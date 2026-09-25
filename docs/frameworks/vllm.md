# vLLM

Generated from `data/recipes/` - do not edit by hand.

Serves Hugging Face models with high throughput on NVIDIA and AMD GPUs, including the FP8, AWQ, GPTQ and compressed-tensors checkpoints

- Runs on: nvidia, amd; on linux
- Install: `uv pip install vllm --torch-backend=auto`
- Home: <https://github.com/vllm-project/vllm>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `vllm/serve` | serve | llm | docs | 0.30.0 |

## `vllm/serve`

Serve step; from the documentation; not yet run here at 0.30.0.

Install: uv pip install vllm --torch-backend=auto

```bash
vllm serve <model> --max-model-len 8192 --gpu-memory-utilization 0.92 --port 8000
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | Hub id or local directory |
| `quantization` | str |  | leave empty for pre-quantized checkpoints (AWQ, GPTQ, FP8), which name their own method |
| `max_model_len` | int | 8192 | prompt plus output |
| `gpu_memory_utilization` | float | 0.92 |  |
| `port` | int | 8000 |  |

- vLLM 0.30 moved bitsandbytes and GGUF out of the main package: uv pip install vllm-bnb-plugin or vllm-gguf-plugin first
- --quantization values are listed in vllm/model_executor/layers/quantization/__init__.py (awq, gptq, fp8, compressed-tensors, mxfp4, ...)
- --gpu-memory-utilization is per instance and defaults to 0.92: vLLM preallocates that share for the KV cache

Source: <https://docs.vllm.ai/en/stable/cli/serve/>
