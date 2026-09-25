# trl

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `trl/sft` | finetune | llm | docs | 1.14.0 |

## `trl/sft`

Finetune step; from the documentation; not yet run here at 1.14.0.

Install: pip install "trl[peft]" bitsandbytes

```bash
trl sft --model_name_or_path <model> --dataset_name trl-lib/Capybara --load_in_4bit --use_peft --lora_r 16 --lora_alpha 32 --dtype bfloat16 --output_dir outputs
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `dataset` | str | trl-lib/Capybara | a Hub dataset of conversations; replace with yours |
| `load_in_4bit` | bool | True | QLoRA; false for 16-bit LoRA |
| `lora_r` | int | 16 |  |
| `lora_alpha` | int | 32 |  |
| `dtype` | enum | bfloat16 | the default, float32, also becomes the 4-bit compute dtype |
| `output_dir` | path | outputs |  |

- --load_in_4bit only works with --use_peft; drop it for 16-bit LoRA
- the flag is --dtype, not --torch_dtype; flags checked against trl/trainer/model_config.py at v1.14.0
- the result is a LoRA adapter: merge it into the base model before converting to GGUF
- TRL 1.14 wants datasets>=4.7, which unsloth's pins exclude: keep them in separate environments

Source: <https://huggingface.co/docs/trl/clis>
