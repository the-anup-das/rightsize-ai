# Sentence Transformers

Generated from `data/recipes/` - do not edit by hand.

Exports embedding models to ONNX and quantizes them to int8 for fast CPU inference

- Runs on: cpu
- Writes: onnx
- Install: `pip install "sentence-transformers[onnx]"` (5.7.0, not 6.x: sentence-transformers 6 needs transformers 5, and optimum-onnx 0.1.0, which the onnx extra brings, pins transformers below 4.58, so 6.1.0 with the extra does not resolve)
- Home: <https://sbert.net>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `sentence-transformers/onnx-int8` | quantize | embedding | run | 5.7.0 |

## `sentence-transformers/onnx-int8`

Quantize step; run end to end at 5.7.0.

`rightsize quantize MODEL` runs it with `--to onnx-int8`.

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

- load: SentenceTransformer(output_dir, backend='onnx', model_kwargs={'file_name': 'onnx/model_quint8_avx2.onnx'}); avx2 stores unsigned weights (quint8), arm64 and the avx512 targets signed (onnx/model_qint8_<target>.onnx)
- output_dir keeps the fp32 export (onnx/model.onnx) the int8 file is made from, about four times its size
- dynamic int8 is for CPUs and likely slower on a GPU
- the [onnx] extra needs sentence-transformers 5.7 with transformers 4.57: optimum-onnx pins transformers<4.58 and sentence-transformers 6 needs 5
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to onnx-int8): all-MiniLM-L6-v2 in 45 s, a 23.0 MB int8 file with the embedding quantized too; its embeddings are within cosine 0.991-0.993 of the fp32 export's

Source: <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>
