# llama.cpp

Generated from `data/recipes/` - do not edit by hand.

Converts Hugging Face models to GGUF, computes importance matrices, quantizes to every GGUF type, measures what quantization cost (perplexity and KL divergence), and serves an OpenAI-compatible API, on CPUs and GPUs from every vendor

- Runs on: any hardware; on linux, windows, macos
- Writes: gguf
- Install: `rightsize tools install llama.cpp` (pinned release binaries plus the converter from the same tag; the conversion step also needs torch and transformers: pip install "rightsize[llamacpp]")
- Home: <https://github.com/ggml-org/llama.cpp>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `llama.cpp/convert` | convert | llm | run | b11177 |
| `llama.cpp/imatrix` | calibrate | llm | run | b11177 |
| `llama.cpp/kld-base` | evaluate | llm | run | b11177 |
| `llama.cpp/kld-eval` | evaluate | llm | run | b11177 |
| `llama.cpp/quantize` | quantize | llm | run | b11177 |
| `llama.cpp/server` | serve | llm | help | b11177 |

## `llama.cpp/convert`

Convert step; run end to end at b11177.

Install: pip install "rightsize[llamacpp]"   # torch + transformers for convert_hf_to_gguf.py

```bash
<python> <convert_script> <model_dir> --outfile <outfile> --outtype auto
```

| input | type | default | notes |
|---|---|---|---|
| `python` | path | required | interpreter with torch |
| `convert_script` | path | required | path to convert_hf_to_gguf.py |
| `model_dir` | path | required | local snapshot of the Hub repo |
| `outfile` | path | required |  |
| `outtype` | enum | auto | auto, f16, bf16, f32, q8_0 |

- auto keeps the source dtype (bf16 stays bf16); f16 is safest for older backends
- runs on CPU; needs roughly 2x the model size in RAM

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/convert_hf_to_gguf.py>

## `llama.cpp/imatrix`

Calibrate step; run end to end at b11177.

Install: download a release from https://github.com/ggml-org/llama.cpp/releases (llama-imatrix)

```bash
<imatrix_bin> -m <model_gguf> -f <calibration_file> -o <output_file> -ngl all -c 512
```

| input | type | default | notes |
|---|---|---|---|
| `imatrix_bin` | path | required |  |
| `model_gguf` | path | required | 16-bit GGUF |
| `calibration_file` | path | required | plain text; ~100-300 KB of mixed-domain text |
| `output_file` | path | required |  |
| `gpu_layers` | str | all |  |
| `ctx` | int | 512 |  |
| `chunks` | int |  | limit processed chunks for speed |

- calibration.method: activation importance over the calibration text; calibration.samples: all chunks unless limited
- required for IQ types; improves K-quants slightly

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/tools/imatrix/README.md>

## `llama.cpp/kld-base`

Evaluate step; run end to end at b11177.

Install: download a release from https://github.com/ggml-org/llama.cpp/releases (llama-perplexity)

```bash
<perplexity_bin> -m <model_gguf> -f <eval_file> --kl-divergence-base <logits_file> -ngl all -c 512
```

| input | type | default | notes |
|---|---|---|---|
| `perplexity_bin` | path | required |  |
| `model_gguf` | path | required | the 16-bit reference |
| `eval_file` | path | required | plain text |
| `logits_file` | path | required | where the reference logits are saved |
| `gpu_layers` | str | all |  |
| `ctx` | int | 512 |  |
| `chunks` | int |  |  |

- saves the reference logits once; every quant is then compared against this file

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/tools/perplexity/README.md>

## `llama.cpp/kld-eval`

Evaluate step; run end to end at b11177.

Install: download a release from https://github.com/ggml-org/llama.cpp/releases (llama-perplexity)

```bash
<perplexity_bin> -m <model_gguf> -f <eval_file> --kl-divergence-base <logits_file> --kl-divergence -ngl all -c 512
```

| input | type | default | notes |
|---|---|---|---|
| `perplexity_bin` | path | required |  |
| `model_gguf` | path | required | the quantized file under test |
| `eval_file` | path | required |  |
| `logits_file` | path | required | reference logits from kld-base |
| `gpu_layers` | str | all |  |
| `ctx` | int | 512 |  |
| `chunks` | int |  |  |

- reports mean KLD, KLD percentiles, top-1 agreement and PPL delta vs the reference

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/tools/perplexity/README.md>

## `llama.cpp/quantize`

Quantize step; run end to end at b11177.

Install: download a release from https://github.com/ggml-org/llama.cpp/releases (llama-quantize)

```bash
<quantize_bin> <input_gguf> <output_gguf> Q2_K
```

| input | type | default | notes |
|---|---|---|---|
| `quantize_bin` | path | required |  |
| `input_gguf` | path | required | 16-bit GGUF from the convert step |
| `output_gguf` | path | required |  |
| `quant` | enum | required | Q2_K, Q2_K_S, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, F16, BF16 |
| `imatrix` | path |  | importance matrix; required for IQ types |
| `pure` | bool | False | disable k-quant mixes |
| `threads` | int |  |  |

- quantize from 16-bit or 32-bit only; requantizing a quantized file loses quality
- K-quant file types are mixes: Q4_K_M keeps some attention/ffn tensors and output at Q6_K

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/tools/quantize/README.md>

## `llama.cpp/server`

Serve step; flags checked against the tool's --help at b11177.

Install: download a release from https://github.com/ggml-org/llama.cpp/releases (llama-server), or rightsize tools install llama.cpp

```bash
<server_bin> -m <model_gguf> -c 8192 -ngl all -fa auto --host 127.0.0.1 --port 8080
```

| input | type | default | notes |
|---|---|---|---|
| `server_bin` | path | required |  |
| `model_gguf` | path | required | the quantized GGUF to serve |
| `ctx` | int | 8192 | context length the plan was sized for |
| `gpu_layers` | str | all | layers on the GPU; all |
| `host` | str | 127.0.0.1 |  |
| `port` | int | 8080 |  |
| `flash_attn` | enum | auto | auto, on, off |

- serves an OpenAI-compatible API at http://host:port/v1; the chat template comes from the GGUF (--jinja is on by default)
- keep --host 127.0.0.1 unless you mean to expose it; add --api-key before binding to 0.0.0.0
- -c is the context the plan was sized for; a larger one needs more KV cache than the plan counted

Source: <https://github.com/ggml-org/llama.cpp/blob/b11177/tools/server/README.md>
