# MLX LM

Generated from `data/recipes/` - do not edit by hand.

Quantizes, fine-tunes (LoRA, and QLoRA on a model it quantized first) and serves LLMs on Apple silicon with Apple's MLX; the default trainer on a Mac

- Runs on: apple; on macos
- Writes: mlx
- Install: `pip install mlx-lm`
- Fine-tunes: lora, qlora; the default trainer on apple
- Home: <https://github.com/ml-explore/mlx-lm>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `mlx-lm/convert` | quantize | llm | docs | 0.31.3 |
| `mlx-lm/fuse` | export | llm | docs | 0.31.3 |
| `mlx-lm/lora` | finetune | llm | docs | 0.31.3 |
| `mlx-lm/server` | serve | llm | docs | 0.31.3 |

## `mlx-lm/convert`

Quantize step; from the documentation; not yet run here at 0.31.3.

`rightsize quantize MODEL` runs it with `--to mlx-4bit` or `--to mlx-8bit`.

Install: pip install mlx-lm

```bash
mlx_lm.convert --hf-path <model> -q --q-bits 4 --mlx-path mlx_model
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `bits` | enum | 4 | 2, 3, 4, 5, 6, 8 |
| `mlx_path` | path | mlx_model | must not exist yet |

- affine quantization with group size 64 by default (--q-group-size); --q-mode also takes mxfp4, nvfp4 and mxfp8
- flags checked against mlx_lm/convert.py at v0.31.3; --model is an alias of --hf-path

Source: <https://github.com/ml-explore/mlx-lm/blob/main/README.md>

## `mlx-lm/fuse`

Export step; from the documentation; not yet run here at 0.31.3.

Install: pip install mlx-lm

```bash
mlx_lm.fuse --model <model> --adapter-path adapters --save-path fused_model --dequantize
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | the model the adapter was trained on (the 4-bit MLX one for QLoRA) |
| `adapter_dir` | path | adapters |  |
| `merged_dir` | path | fused_model |  |

- --dequantize writes 16-bit weights, which the GGUF converter reads; without it a QLoRA fuse stays 4-bit MLX

Source: <https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/fuse.py>

## `mlx-lm/lora`

Finetune step; from the documentation; not yet run here at 0.31.3.

Install: pip install mlx-lm

```bash
mlx_lm.lora --model <model> --train --data data --iters 1000 --batch-size 4 --adapter-path adapters
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | Hub id or local path; a quantized model trains QLoRA |
| `data` | path | data | a directory with train.jsonl (and valid.jsonl), or a Hub dataset |
| `iters` | int | 1000 |  |
| `batch_size` | int | 4 |  |
| `adapter_dir` | path | adapters |  |

- QLoRA means a quantized model: convert with mlx_lm.convert -q first, or start from a 4-bit mlx-community repo
- rank, scale and dropout have no flags: set lora_parameters in a YAML passed with -c (the key is scale, not alpha)
- --fine-tune-type dora or full switches the method; --num-layers (default 16) sets how many layers train
- the result is an adapter: fuse it into a 16-bit model before converting to GGUF

Source: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>

## `mlx-lm/server`

Serve step; from the documentation; not yet run here at 0.31.3.

Install: pip install mlx-lm

```bash
mlx_lm.server --model <model> --port 8080
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | Hub id or the converted directory |
| `port` | int | 8080 |  |

- binds 127.0.0.1 by default (--host); --adapter-path serves a trained adapter on top of the model

Source: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>
