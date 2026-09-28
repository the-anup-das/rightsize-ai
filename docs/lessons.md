# What building this taught us

Rightsize promises numbers that are honest and sourced. Between 2026-09-24 and 2026-09-28 the
first build-out (F1 to F9) tested that promise against real runs on one machine - an RTX 4070
Ti SUPER, 16 GB, Windows 11 - and the runs corrected the estimates more often than they
confirmed them. Each plan in `docs/plans/` records its own findings item by item. This page
keeps the lessons that cut across features, each with the measurement that taught it and
where the fix lives, so the next contributor does not pay for them again.

## An estimate that matches can still be wrong

**Two errors cancelled, twice.** The first `quantize --to fp8` run predicted 0.752 GB and
measured 0.752 GB. It was luck: the embedding stays 16-bit (the size model counted it at 8) and
the checkpoint stores its tied output head a second time (counted once), and at 8 bits the two
cancel to the byte. W4A16 exposed it a run later at +41%. The gpt-oss-20b reading in LM Studio
(+5.2%) was the same shape: MXFP4 sized at 4.25 bits throughout, when llama-quantize writes
everything but the routed experts as Q8_0, cancelled against an embedding counted in VRAM that
llama.cpp keeps in RAM. A total that matches is not evidence until each term has been checked
on its own. `execution/quantize_to.py` now sizes the embedding and head apart, and
`fit/llm.py` splits MXFP4_MOE; `tests/test_quantize_to.py` and `tests/test_llm_embedding.py`
hold the measurements.

**A "first estimate" is the largest error in the model until it is refit.** The llama.cpp
overhead constant was a hand figure, 0.75 GB + 2% of the weights, and labelled as such in
`data/runtimes/overheads.yaml`. Measured over twelve `llama-server` loads (six GGUFs at two
contexts), it was +54% on average and +106% at worst for small models; the fit is 0.27 GB + 1%,
mean error 0.0%, worst -7.5%. About 0.23 GB of it is the CUDA context, the same in every run.
Ollama's and LM Studio's constants are still first estimates and say so in the file.

**Format names hide mixes.** Q4_K_M keeps some tensors at Q6_K. "MXFP4" in a gpt-oss file
name is llama.cpp's MXFP4_MOE: MXFP4 for the experts, Q8_0 for the rest (12.10 GB of tensors
against the 11.11 GB that 4.25 bits gives). OpenVINO's `--weight-format int4` is int4_asym in
groups of 128 with a u4 zero point, 4.156 bits, and NNCF puts embeddings and the last layer at
int8. bitsandbytes and llm-compressor quantize Linear layers only. The documentation rarely
says any of this; the tool's source at the pinned tag, or the tensors it wrote, always does.
Every such rule in `data/` cites the file and tag it was read from.

**The GiB/GB mix-up is easy to make and expensive.** Device memory came in GiB and model sizes
in decimal GB; a 16.7 GB model on a 16 GiB (17.18 GB) card was judged against 16.0 and came
back `no_fit`. `Device` now names the unit in every field (`memory_gib`, `memory_gb`), and a test keeps a
model sized between the two readings fitting.

## Where the bytes really are

**llama.cpp keeps the input embedding in system RAM, always.** `src/llama-model.cpp` puts the
input layer on the CPU. A model with an output head of its own therefore needs that much less
VRAM (175 MB for Qwen3-1.7B at Q4_K_M); a model whose head is tied to the embedding keeps a
copy on the GPU as the head, and needs the RAM as well. Its tensor type follows llama-quantize's
rules - Q6_K when tied, the file type's own otherwise, Q8_0 when the hidden size does not
divide the block - which are in `data/runtimes/llama.cpp/embedding.yaml`. The rule reproduced
the token_embd bytes of every GGUF fixture, and llama.cpp's own CPU and CUDA buffers on load
to 0.02 MB, for six of seven files; the seventh is a publisher's own tensor mix, which only a
header read can size. On unified memory there is nothing to move, and the rule does not apply.

**The same model needs different memory depending on what the tool is doing.** A server keeps
logits only for the tokens it samples, so its compute buffer is 23-64 MiB. `llama-perplexity`
and `llama-imatrix` keep every token's logits and run `n_batch / ctx` sequences at once - four
at `-c 512` - so a perplexity pass on Qwen3-1.7B Q4_K_M takes 1.88 GB where a server at the
same context needs about 1.4 GB. `estimate(..., all_logits=True)` sizes the passes
(1.89 GB predicted). One estimate cannot serve both, and the KL gate's own predictions were
wrong until it asked for the right one.

**A tied head stored twice counts or does not, depending on who loads it.** Qwen3-0.6B and
1.7B store `lm_head.weight` although their configs tie it to the embedding. transformers ties
it away at load; llama.cpp's converter keeps it as `output.weight`; after a fine-tune only one
copy survives, because trainers save through transformers or MLX. F1 records the fact as
`tied_head_stored` and each consumer decides: `fit.finetune.merged()` drops the copy for every
step after a fine-tune, which turned a 0.799 GB prediction for a Q8_0 that measured 0.639 GB
into 0.633.

**Fine-tuning has three honest memory figures, and only one is the need.** Unsloth QLoRA on
Qwen3-0.6B (batch 2, 2048 tokens) allocated at most 1.80 GB, PyTorch reserved 3.43 GB, and the
card rose 4.17 GB. The caching allocator takes what a roomy card offers and gives it back when
memory is short, so an estimate fitted to the rise would fail on a full card. The estimate
models what is allocated; the parts had no term for the loss's logits (batch x sequence x a
152k vocabulary), which is why a 0.6B model allocated twice what they predicted. With
0.7 x that in bf16 - Unsloth chunks its loss - the parts are within 4% on two models.

**`nvidia-smi memory.used` includes the desktop.** About 1.5 GB on this machine. A VRAM
figure is only the rise over a baseline taken before the step; one reading without it produced
a finding that had to be amended. `_VramSampler` in `execution/llamacpp.py` measures the rise.

**KV cache is not one formula.** `2 x layers x kv_heads x head_dim x ctx x bytes` is wrong by
orders of magnitude for MLA (DeepSeek-V3 at 128K is about 9 GB, not the 655 GB the formula
gives), for sliding-window layers, and for hybrid models whose recurrent layers keep a fixed
state. `data/runtimes/llama.cpp/kv_cache.yaml` holds each layout as llama.cpp allocates it,
checked against `llama-fit-params` with 0.0% worst disagreement.

**Derive a number rather than copy it when the primary facts are public.** Nothing
redistributable publishes GPU memory bandwidth, but bus width, memory type and clock are on
Wikipedia and the arithmetic is exact for GDDR and LPDDR. It also catches errors: Wikipedia's
own bandwidth column says 336 GB/s for the RTX 3060 12 GB where its bus width and clock give
the 360 NVIDIA publishes. `scripts/ingest_bandwidth.py` refuses to write if it disagrees by
more than 2% with a hand-checked figure.

## Toolkits

**One environment per toolkit is forced, not preferred.** Unsloth pins `trl<=0.24.0`,
`transformers<=5.5.0` and `datasets<4.4`; sentence-transformers 6 needs transformers 5 while
its own ONNX extra, optimum-onnx 0.1.0, pins transformers below 4.58, so
`sentence-transformers[onnx]==6.1.0` cannot install at all (5.7.0 does). Extras of one
package cannot hold that; `rightsize tools install NAME` gives each toolkit a uv environment
under `.tools/`, and `framework.yaml` pins the version that ran and notes what resolved with
it. `verified: run` on a recipe means exactly that, and `tests/test_registry.py` pins the set
so adding one is a deliberate edit.

**The install check must prove the capability, not the import.** rightsize's own environment
has transformers on a CPU build of PyTorch for the GGUF converter; `import transformers`
succeeds there and NF4 quantization then fails. The Transformers toolkit checks
`bitsandbytes` instead.

**Named calibration presets download whole datasets.** llm-compressor's `perfectblend` is
1.5 GB, fetched whole for a 512-row slice; Open-Platypus is 16 MB and also registered. The
W4A16 recipe defaults to the small one and says why.

**Size is not proof of a working output.** Every `--to` format was loaded back with the
toolkit that serves it and asked a question, or its embeddings compared with the fp32
export's (cosine 0.991-0.993). That is what separates "wrote a file of the right size" from
"works", and the recipe notes record what was done.

**What Windows cost.** CTranslate2's wheel ships no cuBLAS, so its GPU path needs
`cublas64_12.dll` on PATH (a CUDA 12 build of PyTorch has it in `torch/lib`). Unsloth
compiles kernels into an `unsloth_compiled_cache` folder under AGPL in whatever directory it
runs from; it appeared in the repository root once, and the import check now runs where the
cache belongs. The console codepage is cp1252, so unencodable output crashed the CLI until it
replaced such characters. HF's cache warns about symlinks on every download.

## Process

**Pipes hide exit codes.** `pytest | tail` let a failing test into a commit, and a
pre-commit scan built as `grep | head` reported a block with nothing to show, because `head`
exits 0 on empty input. Gate on the producer's exit code (`cmd > log; rc=$?`) or on a count,
never on the last stage of a pipe.

**Generated files need a staleness test.** `docs/frameworks/` and `data/schema/` are
generated from the recipes and the data models; `tests/test_framework_docs.py` and
`tests/test_data_schemas.py` fail when they fall behind, and did so twice in one day before a
commit could carry a stale page.

**Build one real result through the whole pipeline early.** F8 was built before F4 on
purpose. Quantizing and gating one model end to end found the converter packaging, the GiB/GB
mix-up and an sdist that shipped without data, each of which would have surfaced later as a
user's bug report.

**Data over code held up.** A toolkit adds a `--to` format with one `targets` line in its
recipe; llama.cpp's embedding placement is a YAML file with a schema; a plugin package
shipping a descriptor and a recipe became a default trainer with no change to rightsize
(`tests/test_frameworks.py`), and a plugin format ran through the same runner and size check
as the bundled ones (`tests/test_quantize_to.py`). When a rule had to become code, the code
reads the rule from `data/` rather than restating it.

**Write a re-run to a new folder.** Run directories are evidence; a re-run goes beside the
old one (`runs/fp8-qwen06-b`), never over it, so a corrected finding can be compared with the
one it corrects.

**Every finding here came from one machine.** One GPU, one driver, one operating system.
The constants say so in their `source` notes, and F9's calibration loop exists to replace
them with many machines' readings.
