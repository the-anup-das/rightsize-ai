# Sentence Transformers

Generated from `data/recipes/` - do not edit by hand.

Exports embedding models to ONNX and quantizes them to int8 for fast CPU inference

- Runs on: cpu
- Writes: onnx
- Install: `pip install "sentence-transformers[onnx]"`
- Home: <https://sbert.net>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `sentence-transformers/onnx-int8` | quantize | embedding | docs | 6.1.0 |

## `sentence-transformers/onnx-int8`

Quantize step; from the documentation; not yet run here at 6.1.0.

Install: pip install "sentence-transformers[onnx]"

```python
from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

model = SentenceTransformer("<model>", backend="onnx")
model.save_pretrained("onnx-model")
export_dynamic_quantized_onnx_model(
    model=model, quantization_config="avx2", model_name_or_path="onnx-model")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `output_dir` | path | onnx-model |  |
| `cpu_target` | enum | avx2 | avx2, avx512, avx512_vnni, arm64 |

- load: SentenceTransformer(output_dir, backend='onnx', model_kwargs={'file_name': 'onnx/model_qint8_<target>.onnx'})
- dynamic int8 is for CPUs and likely slower on a GPU
- the [onnx] extra resolves to sentence-transformers 5.7 with transformers 4.57 (optimum-onnx pins transformers<4.58)

Source: <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>
