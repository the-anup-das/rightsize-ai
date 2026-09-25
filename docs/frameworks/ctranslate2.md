# ctranslate2

Generated from `data/recipes/` - do not edit by hand.

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `ctranslate2/convert-whisper` | convert | audio | docs | 4.8.2 |

## `ctranslate2/convert-whisper`

Convert step; from the documentation; not yet run here at 4.8.2.

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

Source: <https://opennmt.net/CTranslate2/guides/transformers.html>
