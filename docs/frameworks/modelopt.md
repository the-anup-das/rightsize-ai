# NVIDIA Model Optimizer

Generated from `data/recipes/` - do not edit by hand.

Quantizes Hugging Face models to FP8, INT8 SmoothQuant, INT4 AWQ and NVFP4 checkpoints that vLLM, SGLang and TensorRT-LLM load

- Runs on: nvidia
- Writes: fp8, int8, int4, nvfp4
- Install: `pip install nvidia-modelopt transformers accelerate datasets` (a pure-Python wheel; the quantizers run in PyTorch, with CUDA kernels compiled on first use where a compiler exists and plain PyTorch otherwise. Quantizing works on any CUDA GPU; serving the FP8 output needs Ada or newer and NVFP4 needs Blackwell, which the serve-time rules say)
- Home: <https://github.com/NVIDIA/Model-Optimizer>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `modelopt/autoquant` | quantize | llm | run | 0.47.0 |
| `modelopt/ptq` | quantize | llm | run | 0.47.0 |

## `modelopt/autoquant`

Quantize step; run end to end at 0.47.0.

`rightsize quantize MODEL` runs it with `--to modelopt-auto`.

Install: pip install nvidia-modelopt transformers accelerate datasets

```python
"""Mixed-precision quantization to a bit budget: Model Optimizer's AutoQuantize scores how
much each layer suffers under each candidate format and keeps the more sensitive layers at
higher precision, or unquantized, so the model as a whole meets effective_bits."""
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"<model>")
model = AutoModelForCausalLM.from_pretrained(r"<model>", dtype=torch.bfloat16).cuda()
rows = load_dataset("garage-bAInd/Open-Platypus", split="train").select(range(64))
batches = []
for row in rows:
    enc = tokenizer(row["output"], return_tensors="pt", truncation=True,
                    max_length=512)
    batch = {k: v.cuda() for k, v in enc.items()}
    batch["labels"] = batch["input_ids"]  # the loss the sensitivity scores come from
    batches.append(batch)

model, search = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=[name.strip() for name in "NVFP4_DEFAULT_CFG,FP8_DEFAULT_CFG".split(",")],
    data_loader=batches,
    forward_step=lambda m, batch: m(**batch),
    loss_func=lambda output, batch: output.loss,
    num_calib_steps=len(batches),
    num_score_steps=32,
    method="gradient",
    disabled_layers=["*lm_head*"],
    verbose=True,
)
mto.save(model, r"modelopt_state.pth")
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=r"model-modelopt-auto")
tokenizer.save_pretrained(r"model-modelopt-auto")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `effective_bits` | float | 4.8 | the average bits per weight over the quantized layers; rightsize quantize sets it to what the device has room for (fit.bits_that_fit) unless --set names one |
| `formats` | str | NVFP4_DEFAULT_CFG,FP8_DEFAULT_CFG | the presets the search picks from per layer, comma-separated (modelopt.torch.quantization.config.choices); leaving a layer unquantized is always a choice |
| `method` | enum | gradient | how a layer's sensitivity to each format is scored |
| `dataset` | str | garage-bAInd/Open-Platypus | a Hub dataset whose rows have the text column; 16 MB |
| `text_column` | str | output |  |
| `num_samples` | int | 64 | rows for calibration; the docs suggest 512 for a final model |
| `num_score_steps` | int | 32 | rows for the sensitivity scores; the docs suggest 128, and this phase is most of the time |
| `max_seq_length` | int | 512 |  |
| `output_dir` | path | model-modelopt-auto |  |
| `state_file` | path | modelopt_state.pth | the searched, calibrated quantizer state; the gate restores it onto the 16-bit model |

- --to modelopt-auto: rightsize quantize computes effective_bits from the device and --ctx (usable memory less the KV cache and the runtime's overhead, the embedding kept at 16 bits) unless --set effective_bits=N names one
- the example recipe pairs NVFP4 with FP8 (general/auto_quantize/nvfp4_fp8_at_5p4bits); NVFP4's simulated quantization needs the MX CUDA kernel, so on a machine without nvcc pass --set formats=FP8_DEFAULT_CFG,INT4_AWQ_CFG
- the search is a forward and backward pass per scored batch per candidate format: minutes on a 0.6B model with the defaults here, longer with the documented 512 and 128
- effective_bits averages over the Linear layers with lm_head counted (kept 16-bit) and the embedding outside; a tied head is a Linear whose tensor is the embedding, so for a tied model the export is exactly that average times every parameter but the stored duplicate (fit.bits_that_fit's bits_linear, and format_weights_gb over=linear)
- %s (rightsize quantize --to modelopt-auto --eval --device 'RTX 3060 12GB' --ctx 57344 --set formats=FP8_DEFAULT_CFG,INT4_AWQ_CFG, 2026-09-28): Qwen3-1.7B, budget 11.7 bits for the 2.517 GB that card leaves after the KV cache; the search (99 s, 7.7 GB of VRAM, 64 rows, 32 scored) put 66 layers at FP8, kept 46 at 16 bits and one at INT4 AWQ, 11.70 effective bits, and wrote 2.516 GB; gate pass, KLD 0.010, top-1 0.956, perplexity 17.16 against 17.11, where every uniform 4-bit format of this model failed. Qwen3-0.6B on the same card at 64k context has room for 16 bits, and the search left every layer as it was (KLD 0)

Source: <https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/examples/hf_ptq/README.md>

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
