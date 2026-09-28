# TRL

Generated from `data/recipes/` - do not edit by hand.

Hugging Face's training library: supervised fine-tuning with LoRA or QLoRA through PEFT and bitsandbytes, from a command line

- Runs on: nvidia, amd, intel
- Writes: safetensors
- Install: `pip install "trl[peft]" bitsandbytes`
- Fine-tunes: lora, qlora; not a default trainer; choose it by name
- Home: <https://github.com/huggingface/trl>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `trl/merge` | export | llm | docs | 0.21.0 |
| `trl/sft` | finetune | llm | docs | 1.14.0 |

## `trl/merge`

Export step; from the documentation; not yet run here at 0.21.0.

Install: pip install "trl[peft]"

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    # loads the base named in the adapter's config at 16 bits, not the 4 bits QLoRA trained on
    model = AutoPeftModelForCausalLM.from_pretrained("outputs", dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.save_pretrained("merged")
    AutoTokenizer.from_pretrained("outputs").save_pretrained("merged")


if __name__ == "__main__":
    main()
```

| input | type | default | notes |
|---|---|---|---|
| `output_dir` | path | outputs | where trl sft saved the adapter |
| `merged_dir` | path | merged | the merged 16-bit model the conversion reads |

- merge_and_unload folds the LoRA weights into the base model and drops the adapter layers

Source: <https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-lora-weights-into-the-base-model>

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
