# transformers

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `transformers/bnb-nf4` | quantize | llm | docs | 5.17.0 |

## `transformers/bnb-nf4`

Quantize step; from the documentation; not yet run here at 5.17.0.

Install: pip install --upgrade transformers accelerate bitsandbytes

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "<model>", quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("<model>")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `double_quant` | enum | True | saves about 0.4 bits per parameter |

- quantizes while loading; nothing is written to disk
- transformers 5 removed load_in_4bit= as a from_pretrained argument, and dtype= replaces torch_dtype=

Source: <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>
