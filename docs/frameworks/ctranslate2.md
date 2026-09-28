# CTranslate2

Generated from `data/recipes/` - do not edit by hand.

Converts Whisper and other transformer models to CTranslate2, int8 or float16, the format faster-whisper runs

- Runs on: nvidia, cpu
- Writes: ctranslate2
- Install: `pip install ctranslate2 "transformers[torch]"`
- Home: <https://github.com/OpenNMT/CTranslate2>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `ctranslate2/convert-whisper` | convert | audio | run | 4.8.2 |

## `ctranslate2/convert-whisper`

Convert step; run end to end at 4.8.2.

`rightsize quantize MODEL` runs it with `--to ct2-int8`.

Install: pip install ctranslate2 "transformers[torch]"

```bash
ct2-transformers-converter --model <model> --output_dir whisper-ct2 --quantization int8_float16 --copy_files tokenizer.json preprocessor_config.json
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | a Whisper repo, e.g. openai/whisper-large-v3-turbo |
| `output_dir` | path | whisper-ct2 |  |
| `quantization` | enum | int8_float16 | int8_float16, int8, int8_float32, int8_bfloat16, int16, float16, bfloat16, float32 |

- load it with faster_whisper.WhisperModel(output_dir); compute_type= can convert the stored type again at load
- rightsize estimate MODEL --quant int8 --runtime faster-whisper gives the memory
- the conversion runs on the CPU; on Windows the GPU needs cublas64_12.dll on PATH, which the ctranslate2 wheel does not ship: a CUDA 12 build of PyTorch has it in torch/lib
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to ct2-int8): whisper-tiny in 44 s, a 38.9 MB model.bin; it decodes on the CPU (int8) and on the GPU (int8_float16)

Source: <https://opennmt.net/CTranslate2/guides/transformers.html>
