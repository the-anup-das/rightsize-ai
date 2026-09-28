# NVIDIA Model Optimizer

Generated from `data/recipes/` - do not edit by hand.

Quantizes Hugging Face models to FP8, INT8 SmoothQuant, INT4 AWQ and NVFP4 checkpoints that vLLM, SGLang and TensorRT-LLM load

- Runs on: nvidia
- Writes: fp8, int8, int4, nvfp4
- Install: `pip install nvidia-modelopt transformers accelerate datasets` (a pure-Python wheel; the quantizers run in PyTorch, with CUDA kernels compiled on first use where a compiler exists and plain PyTorch otherwise. Quantizing works on any CUDA GPU; serving the FP8 output needs Ada or newer and NVFP4 needs Blackwell, which the serve-time rules say)
- Home: <https://github.com/NVIDIA/Model-Optimizer>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `modelopt/ptq` | quantize | llm | run | 0.47.0 |

## `modelopt/ptq`

Quantize step; run end to end at 0.47.0.

`rightsize quantize MODEL` runs it with `--to modelopt-fp8` or `--to modelopt-int8-sq` or `--to modelopt-int4-awq` or `--to modelopt-nvfp4`.

Install: pip install nvidia-modelopt transformers accelerate datasets

```python
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIGS = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}
tokenizer = AutoTokenizer.from_pretrained(r"<model>")
model = AutoModelForCausalLM.from_pretrained(r"<model>", dtype=torch.bfloat16).cuda().eval()
rows = load_dataset("garage-bAInd/Open-Platypus", split="train").select(range(128))
batches = [
    tokenizer(row["output"], return_tensors="pt", truncation=True,
              max_length=512).input_ids.cuda()
    for row in rows
]


def forward_loop(m):
    with torch.no_grad():
        for ids in batches:
            m(ids)


model = mtq.quantize(model, CONFIGS["fp8"], forward_loop)
mto.save(model, r"modelopt_state.pth")  # what the gate restores onto a fresh 16-bit model
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=r"model-modelopt")
tokenizer.save_pretrained(r"model-modelopt")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `qformat` | enum | fp8 | the preset: FP8 per-tensor weights and activations, INT8 SmoothQuant, INT4 AWQ (block 128, weights only), NVFP4 (block 16, FP8 scales) |
| `dataset` | str | garage-bAInd/Open-Platypus | a Hub dataset whose rows have the text column; 16 MB |
| `text_column` | str | output |  |
| `num_samples` | int | 128 | calibration rows the activation ranges come from |
| `max_seq_length` | int | 512 |  |
| `output_dir` | path | model-modelopt | the unified Hugging Face checkpoint vLLM, SGLang and TensorRT-LLM load |
| `state_file` | path | modelopt_state.pth | the calibrated quantizer state; the gate restores it onto the 16-bit model |

- the export is Model Optimizer's unified checkpoint: the original tensor names, quantized weights and an hf_quant_config.json; vllm serve --quantization modelopt loads it, as do SGLang and TensorRT-LLM. Transformers does not, which is why the gate restores the quantizer state instead
- the example scripts calibrate on cnn_dailymail by default; Open-Platypus is 16 MB and its output column is plain prose
- presets come from modelopt_recipes/configs/ptq/presets/model/*.yaml (fp8, int8_smoothquant, int4_awq, nvfp4 and 30 more); these four are the ones vLLM serves on the widest set of GPUs
- mtq.NVFP4_MLP_ONLY_CFG keeps attention in higher precision when NVFP4 costs too much accuracy; AutoQuantize (mtq.auto_quantize) mixes FP8 and NVFP4 to an effective_bits target
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to modelopt-FORMAT --eval, 2026-09-28; Model Optimizer 0.47.0 on torch 2.10+cu128): Qwen3-0.6B on 128 Open-Platypus rows. fp8 in 38 s at 1.9 GB of VRAM, 0.752 GB of weights as predicted, gate pass (KLD 0.022, top-1 0.922, perplexity 22.41 against 22.15). nvfp4 in 37 s, 0.559 GB as predicted (block-16 FP8 scales, embedding and head left 16-bit); its gate needs the MX CUDA kernel Model Optimizer builds on first use, which wants nvcc, so it did not run here. int4_awq in 307 s, 0.546 GB against 0.538 predicted (the AWQ pre-quant scales), gate fail (KLD 0.263, top-1 0.750). int8_sq in 40 s, 0.754 GB, gate fail (KLD 0.366, top-1 0.726, perplexity 30.5): per-tensor int8 activations hurt this small model where weight-only int8 (OpenVINO's, KLD 0.005) does not
- the gate restores the saved state onto a fresh 16-bit model with mto.restore, which carries the weights SmoothQuant and AWQ modified as well as the calibrated scales (mto.save writes both)

Source: <https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/examples/hf_ptq/README.md>
