# Sentence Transformers

Generated from `data/recipes/` - do not edit by hand.

Exports embedding models to ONNX and quantizes them to int8 for fast CPU inference

- Runs on: cpu
- Writes: onnx
- Install: `pip install "sentence-transformers[onnx]"` (5.7.0, not 6.x: sentence-transformers 6 needs transformers 5, and optimum-onnx 0.1.0, which the onnx extra brings, pins transformers below 4.58, so 6.1.0 with the extra does not resolve)
- Home: <https://sbert.net>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `sentence-transformers/cosine-eval` | evaluate | embedding | run | 5.7.0 |
| `sentence-transformers/onnx-int8` | quantize | embedding | run | 5.7.0 |

## `sentence-transformers/cosine-eval`

Evaluate step; run end to end at 5.7.0.

Install: pip install "sentence-transformers[onnx]"

```python
"""Cosine similarity between the quantized model's embeddings and the original's, over
lines of a text. An embedding model is judged by the neighbourhoods it keeps, and a
quantization that moves every vector a little moves its neighbours the same way, so the
mean and the worst cosine say how much of the original's ranking survives."""
import json

from sentence_transformers import SentenceTransformer

with open(r"<eval_file>", encoding="utf-8") as fh:
    texts = [line.strip() for line in fh if len(line.strip()) > 40][:200]
base = SentenceTransformer(r"<model>")
candidate = SentenceTransformer(
    r"<candidate>", backend="onnx", model_kwargs={"file_name": r"onnx/model_quint8_avx2.onnx"})
a = base.encode(texts, normalize_embeddings=True)
b = candidate.encode(texts, normalize_embeddings=True)
cos = (a * b).sum(axis=1)
result = {"cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()), "texts": len(texts)}
json.dump(result, open(r"gate.json", "w"), indent=1)
print("gate", json.dumps(result))
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | the model the candidate was exported from |
| `candidate` | path | required | the exported directory |
| `file_name` | str | onnx/model_quint8_avx2.onnx | the quantized ONNX file inside it |
| `eval_file` | path | required | plain text; one embedding per line of it |
| `lines` | int | 200 | lines embedded (the first ones over 40 characters) |
| `gate_file` | path | gate.json |  |

- thresholds in data/quality/gate_thresholds.yaml (embedding_cosine); the base model runs on the CPU through PyTorch in the same environment
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to onnx-int8 --eval, 2026-09-28): all-MiniLM-L6-v2's int8 export, cosine 0.991 mean and 0.982 worst over 200 lines, 42 s: pass

Source: <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>

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
- gate (--eval): pass, cosine 0.991 mean and 0.982 worst over 200 lines of wikitext-2

Source: <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>
