# Unsloth

Generated from `data/recipes/` - do not edit by hand.

LoRA and QLoRA fine-tuning on a single GPU, writing a merged 16-bit model (or a GGUF) at the end; the default trainer on NVIDIA and Intel GPUs

- Runs on: nvidia, amd, intel; on linux, windows; NVIDIA compute capability 7.0+
- Writes: safetensors
- Install: `uv pip install unsloth --torch-backend=auto` (in its own environment: unsloth pins trl<=0.24.0, transformers<=5.5.0 and datasets<4.4. Installed here with uv on Windows: PyTorch 2.11 for CUDA 12.8, Transformers 5.5.0, TRL 0.24.0 and triton-windows came with it, 5.0 GB)
- Fine-tunes: lora, qlora; the default trainer on nvidia, intel and wherever no other framework is
- Home: <https://unsloth.ai>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `unsloth/sft` | finetune | llm | run | 2026.9.11 |

## `unsloth/sft`

Finetune step; run end to end at 2026.9.11.

Install: uv pip install unsloth --torch-backend=auto, in its own environment: unsloth pins trl<=0.24.0, transformers<=5.5.0 and datasets<4.4

```python
from unsloth import FastLanguageModel  # import unsloth before trl and transformers
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="<model>", max_seq_length=2048, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
    dataset = standardize_sharegpt(load_dataset("mlabonne/FineTome-100k", split="train"))

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                         for c in convos]}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=2,
                       gradient_accumulation_steps=4, num_train_epochs=1,
                       max_steps=-1, learning_rate=2e-4, optim="adamw_8bit",
                       output_dir="outputs", report_to="none"))
    trainer.train()
    model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")


if __name__ == "__main__":  # worker processes re-import this file on Windows and macOS
    main()
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | Hub id of the model to fine-tune |
| `load_in_4bit` | enum | True | True for QLoRA, False for 16-bit LoRA |
| `max_seq_length` | int | 2048 |  |
| `lora_r` | int | 16 | 16 or 32, per Unsloth's hyperparameter guide |
| `lora_alpha` | int | 16 | r or 2r |
| `dataset` | str | mlabonne/FineTome-100k | a Hub dataset of conversations; replace with yours |
| `split` | str | train | a slice for a short run, e.g. train[:500] |
| `messages_column` | str | conversations | conversations (ShareGPT from/value) or messages (role/content) |
| `chat_template` | str |  | empty keeps the model's own template; else an Unsloth name such as chatml, llama-3.1, qwen-2.5, gemma-3 |
| `num_epochs` | int | 1 |  |
| `max_steps` | int | -1 | stop after this many optimizer steps; -1 trains num_epochs |
| `output_dir` | path | outputs |  |
| `merged_dir` | path | merged | the merged 16-bit model the llama.cpp steps convert next |

- save as train.py and run it with python where unsloth is installed
- save_pretrained_merged writes the 16-bit model the llama.cpp steps convert next; model.save_pretrained_gguf(dir, tokenizer, quantization_method='q4_k_m') is the one-step alternative, and builds llama.cpp on first use
- use the same chat template at inference as in training
- tokenizer= is Unsloth's name for TRL's processing_class; TRL 0.24 calls the length max_length
- run end to end on Windows with an RTX 4070 Ti SUPER: QLoRA on Qwen3-0.6B, 30 steps in 37 s, then the merge and llama.cpp's convert and Q8_0 (rightsize run)

Source: <https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb>
