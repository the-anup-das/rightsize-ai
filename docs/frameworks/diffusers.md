# diffusers

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `diffusers/quantize-pipeline` | quantize | diffusion | docs | 0.40.0 |

## `diffusers/quantize-pipeline`

Quantize step; from the documentation; not yet run here at 0.40.0.

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
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `components` | str | ["transformer", "text_encoder_2"] | the components to quantize; FLUX: transformer and text_encoder_2 (T5) |
| `offload` | bool | False | enable_model_cpu_offload instead of moving the pipeline to the GPU |

- rightsize estimate MODEL --quant nf4 --te-quant nf4 --offload model sizes this before you download it
- dtype= replaces torch_dtype= (passing both is an error)
- torchao instead: quant_mapping={'transformer': TorchAoConfig(Int8WeightOnlyConfig())}; TorchAoConfig no longer takes strings, and the docs' Int8WeightOnlyConfig(group_size=128, version=2) fails on torchao 0.18

Source: <https://huggingface.co/docs/diffusers/en/quantization/overview>
