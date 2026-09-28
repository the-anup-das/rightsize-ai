# Diffusers

Generated from `data/recipes/` - do not edit by hand.

Loads diffusion pipelines with chosen components quantized (bitsandbytes, torchao, GGUF) through PipelineQuantizationConfig

- Runs on: nvidia, amd, intel
- Writes: bnb
- Install: `pip install -U diffusers transformers accelerate bitsandbytes`
- Home: <https://github.com/huggingface/diffusers>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `diffusers/quantize-pipeline` | quantize | diffusion | run | 0.40.0 |

## `diffusers/quantize-pipeline`

Quantize step; run end to end at 0.40.0.

Install: pip install -U diffusers transformers accelerate bitsandbytes

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipe = DiffusionPipeline.from_pretrained("<model>", quantization_config=quant, dtype=torch.bfloat16)
pipe.to("cuda")
pipe.save_pretrained(r"model-nf4")  # the quantized components keep their quantization config
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `components` | str | ["transformer", "text_encoder_2"] | the components to quantize; FLUX: transformer and text_encoder_2 (T5) |
| `offload` | bool | False | enable_model_cpu_offload instead of moving the pipeline to the GPU |
| `output_dir` | path | model-nf4 | where the pipeline is saved with its quantized components; from_pretrained loads it back 4-bit |

- rightsize estimate MODEL --quant nf4 --te-quant nf4 --offload model sizes this before you download it
- dtype= replaces torch_dtype= (passing both is an error)
- torchao instead: quant_mapping={'transformer': TorchAoConfig(Int8WeightOnlyConfig())}; TorchAoConfig no longer takes strings, and the docs' Int8WeightOnlyConfig(group_size=128, version=2) fails on torchao 0.18
- run on Windows with an RTX 4070 Ti SUPER (rightsize run, a one-step plan, 2026-09-28; diffusers 0.40.0, bitsandbytes 0.50.2): segmind/tiny-sd with components [unet, text_encoder], 315 s of which most was the 1 GB download, 0.98 GB of VRAM, 0.74 GB saved: the UNet 450 MB (its Linear layers NF4; the convolutions, most of a UNet, stay 16-bit), the text encoder 124 MB, the VAE 167 MB untouched. The saved pipeline loads back with Linear4bit modules and makes a 256x256 image in 0.8 s at 0.92 GB
- bitsandbytes quantizes Linear layers only, so a UNet shrinks far less than a DiT: FLUX's transformer is all attention and MLP and is where nf4 pays

Source: <https://huggingface.co/docs/diffusers/en/quantization/overview>
