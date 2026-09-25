# llm-compressor

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `llm-compressor/fp8-dynamic` | quantize | llm | docs | 0.14.0 |
| `llm-compressor/gptq-w4a16` | quantize | llm | docs | 0.14.0 |

## `llm-compressor/fp8-dynamic`

Quantize step; from the documentation; not yet run here at 0.14.0.

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

Source: <https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/oneshot/>

## `llm-compressor/gptq-w4a16`

Quantize step; from the documentation; not yet run here at 0.14.0.

Install: pip install llmcompressor

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

oneshot(
    model="<model>",
    dataset="perfectblend",
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
| `dataset` | str | perfectblend | a name llm-compressor registers (perfectblend, ultrachat_200k, open_platypus, wikitext, ...) |
| `split` | str | train[:512] |  |
| `num_samples` | int | 512 |  |
| `max_seq_length` | int | 2048 |  |
| `output_dir` | path | model-W4A16-G128 |  |

- int4 weights, group 128, 16-bit activations, calibrated on the dataset; serve the directory with vllm serve
- dataset and split follow examples/quantization_w4a16/llama3_example.py at 0.14.0; other data: pass a datasets.Dataset with a text column

Source: <https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/quantization_w4a16/llama3_example.py>
