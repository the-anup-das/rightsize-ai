# axolotl

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `axolotl/qlora` | finetune | llm | docs | 0.19.0 |

## `axolotl/qlora`

Finetune step; from the documentation; not yet run here at 0.19.0.

Install: uv pip install --no-build-isolation "axolotl[deepspeed]", after installing PyTorch for your CUDA or ROCm version

```yaml
base_model: <model>
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    field_messages: conversations
    message_property_mappings:
      role: from
      content: value
sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 1
learning_rate: 0.0002
optimizer: paged_adamw_32bit
bf16: auto
output_dir: ./outputs/qlora-out
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `adapter` | enum | qlora | qlora, lora |
| `load_in_4bit` | bool | True | true with qlora, false with lora |
| `dataset` | str | mlabonne/FineTome-100k | a Hub dataset of conversations; replace with yours |
| `lora_r` | int | 16 |  |
| `lora_alpha` | int | 32 |  |
| `sequence_len` | int | 2048 |  |
| `micro_batch_size` | int | 2 |  |
| `grad_accum` | int | 4 |  |
| `num_epochs` | int | 1 |  |
| `learning_rate` | float | 0.0002 | required by axolotl |
| `output_dir` | path | ./outputs/qlora-out |  |

- save as qlora.yml and run: axolotl train qlora.yml
- field_messages and message_property_mappings read ShareGPT data (from/value); drop them for a messages column of role/content
- the result is a LoRA adapter: merge it into the base model before converting to GGUF

Source: <https://docs.axolotl.ai/docs/config-reference.html>
