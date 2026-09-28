# LLM Compressor

Generated from `data/recipes/` - do not edit by hand.

Quantizes models to the compressed-tensors checkpoints vLLM and SGLang load: FP8 without calibration data, and GPTQ W4A16 with a small calibration set

- Runs on: nvidia
- Writes: compressed-tensors
- Install: `pip install llmcompressor`
- Home: <https://github.com/vllm-project/llm-compressor>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `llm-compressor/awq-w4a16` | quantize | llm | run | 0.14.0 |
| `llm-compressor/fp8-dynamic` | quantize | llm | run | 0.14.0 |
| `llm-compressor/gptq-w4a16` | quantize | llm | run | 0.14.0 |

## `llm-compressor/awq-w4a16`

Quantize step; run end to end at 0.14.0.

`rightsize quantize MODEL` runs it with `--to awq`.

Install: pip install llmcompressor

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

oneshot(
    model="<model>",
    dataset="open_platypus",
    splits="train[:512]",
    recipe=[
        # AWQ scales the channels the calibration activations mark as salient, then the
        # quantizer rounds them: the modifier smooths, it does not quantize on its own
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"]),
    ],
    max_seq_length=512,
    num_calibration_samples=256,
    output_dir="model-AWQ-W4A16-ASYM",
)
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `dataset` | str | open_platypus | a name llm-compressor registers (open_platypus, perfectblend, ultrachat_200k, wikitext, ...) |
| `split` | str | train[:512] |  |
| `num_samples` | int | 256 | calibration samples the activation statistics come from |
| `max_seq_length` | int | 512 |  |
| `output_dir` | path | model-AWQ-W4A16-ASYM |  |

- int4 asymmetric weights, group 128, 16-bit activations, with AWQ's activation-aware scaling; serve the directory with vllm serve
- the example's dataset is perfectblend (1.5 GB fetched whole); open_platypus is 16 MB and also registered
- 0.14 moved AWQModifier to llmcompressor.modifiers.transform.awq and split it from the quantizer; the old llmcompressor.modifiers.awq import still works with a deprecation warning
- AWQ needs a mapping from each norm to the layers it feeds; llm-compressor ships them for the common architectures and infers them for the rest (llmcompressor/modifiers/transform/awq/mappings.py)
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to awq --eval, 2026-09-28): Qwen3-0.6B in 4.5 min on 256 Open-Platypus samples, gate fail (KLD 0.199, top-1 0.778, perplexity 25.26 against 22.15), between NF4's 0.193 and GPTQ's 0.228; Qwen3-1.7B in 306 s at 5.24 GB of VRAM, 1.355 GB of weights as predicted, gate warn (KLD 0.133, top-1 0.850, perplexity 17.14 against 17.11), where llama.cpp's Q4_K_M of the same model scores 0.059

Source: <https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/awq/llama_example.py>

## `llm-compressor/fp8-dynamic`

Quantize step; run end to end at 0.14.0.

`rightsize quantize MODEL` runs it with `--to fp8`.

Install: pip install llmcompressor

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model="<model>",
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="model-FP8-Dynamic",
)
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `output_dir` | path | model-FP8-Dynamic |  |

- no calibration data: FP8 weights with dynamic per-token activation scales; serve the directory with vllm serve
- from llmcompressor.transformers import oneshot is gone in 0.14
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to fp8): Qwen3-0.6B in 12 s once downloaded (392 s with the 1.5 GB download), at most 1.06 GB of VRAM; 752.4 MB of weights, the embedding still BF16; Transformers loads it back through compressed-tensors and it answers
- gate (--eval) on Qwen3-0.6B: pass, KLD 0.020, top-1 agreement 0.925, perplexity 22.50 against the 16-bit model's 22.15

Source: <https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/oneshot/>

## `llm-compressor/gptq-w4a16`

Quantize step; run end to end at 0.14.0.

`rightsize quantize MODEL` runs it with `--to w4a16`.

Install: pip install llmcompressor

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

oneshot(
    model="<model>",
    dataset="open_platypus",
    splits="train[:512]",
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="model-W4A16-G128",
)
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `dataset` | str | open_platypus | a name llm-compressor registers (open_platypus, perfectblend, ultrachat_200k, wikitext, ...) |
| `split` | str | train[:512] |  |
| `num_samples` | int | 512 |  |
| `max_seq_length` | int | 2048 |  |
| `output_dir` | path | model-W4A16-G128 |  |

- int4 weights, group 128, 16-bit activations, calibrated on the dataset; serve the directory with vllm serve
- split and sample count follow examples/quantization_w4a16/llama3_example.py at 0.14.0; other data: pass a datasets.Dataset with a text column
- open_platypus (garage-bAInd/Open-Platypus) is a 16 MB download; the example's perfectblend is 1.5 GB and ultrachat_200k 1.6 GB, fetched whole even for a 512-row slice
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to w4a16): Qwen3-0.6B on 512 Open-Platypus samples in 144 s, at most 1.71 GB of VRAM; 538.5 MB of weights, 220 MB of them the int4 layers and 311 MB the BF16 embedding; loads back and answers
- gate (--eval) on Qwen3-0.6B: fail, KLD 0.228, top-1 0.777, perplexity 25.59 against 22.15. llama.cpp's own Q4_K_M of the same model scores 0.115 (warn) and its Q4_0 0.219: four bits is too few for a 0.6B model whichever toolkit rounds it; these defaults are for 7B-class models, where W4A16 costs about 0.02-0.05

Source: <https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/quantization_w4a16/llama3_example.py>
