# Transformers

Generated from `data/recipes/` - do not edit by hand.

Loads a model quantized on the fly to 4-bit NF4 (or 8-bit) with bitsandbytes, and can save the quantized checkpoint

- Runs on: nvidia, amd, intel
- Writes: bnb
- Install: `pip install --upgrade transformers accelerate bitsandbytes`
- Home: <https://github.com/huggingface/transformers>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `transformers/bnb-nf4` | quantize | llm | run | 5.17.0 |

## `transformers/bnb-nf4`

Quantize step; run end to end at 5.17.0.

`rightsize quantize MODEL` runs it with `--to nf4`.

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
model.save_pretrained("model-bnb-nf4")  # loads back 4-bit, quantization config included
tokenizer.save_pretrained("model-bnb-nf4")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `double_quant` | enum | True | saves about 0.4 bits per parameter |
| `output_dir` | path | model-bnb-nf4 | where the 4-bit checkpoint is saved |

- quantizes while loading, then saves the 4-bit checkpoint; from_pretrained on output_dir loads it back quantized
- transformers 5 removed load_in_4bit= as a from_pretrained argument, and dtype= replaces torch_dtype=
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to nf4): Qwen3-0.6B in 10 s, at most 1.34 GB of VRAM; 538.9 MB of weights with the embedding and the tied head left in BF16; loads back 4-bit and answers

Source: <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>
