# Transformers

Generated from `data/recipes/` - do not edit by hand.

Loads a model quantized on the fly to 4-bit NF4 (or 8-bit) with bitsandbytes, and can save the quantized checkpoint

- Runs on: nvidia, amd, intel
- Writes: bnb
- Install: `pip install --upgrade transformers accelerate bitsandbytes`
- Home: <https://github.com/huggingface/transformers>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `transformers/bnb-nf4` | quantize | llm | run | 5.17.0 |
| `transformers/kld-eval` | evaluate | llm | run | 5.17.0 |

## `transformers/bnb-nf4`

Quantize step; run end to end at 5.17.0.

`rightsize quantize MODEL` runs it with `--to nf4`.

Install: pip install --upgrade transformers accelerate bitsandbytes

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "<model>", quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("<model>")
model.save_pretrained("model-bnb-nf4")  # loads back 4-bit, quantization config included
tokenizer.save_pretrained("model-bnb-nf4")
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `double_quant` | enum | True | saves about 0.4 bits per parameter |
| `output_dir` | path | model-bnb-nf4 | where the 4-bit checkpoint is saved |

- quantizes while loading, then saves the 4-bit checkpoint; from_pretrained on output_dir loads it back quantized
- transformers 5 removed load_in_4bit= as a from_pretrained argument, and dtype= replaces torch_dtype=
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to nf4): Qwen3-0.6B in 10 s, at most 1.34 GB of VRAM; 538.9 MB of weights with the embedding and the tied head left in BF16; loads back 4-bit and answers
- gate (--eval) on Qwen3-0.6B: fail, KLD 0.193, top-1 0.781, perplexity 25.25 against 22.15, about what llama.cpp's Q4_0 does to the same model (0.219); a 0.6B model wants 8 bits, and a 7B-class model takes NF4 with far less damage

Source: <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>

## `transformers/kld-eval`

Evaluate step; run end to end at 5.17.0.

Install: pip install --upgrade transformers accelerate

```python
"""Mean KL divergence, top-1 agreement and perplexity of a quantized checkpoint against the
16-bit model it came from, over windows of a text: what llama-perplexity --kl-divergence
measures, for checkpoints Transformers (or Optimum Intel) loads. Like llama.cpp, it scores
the second half of each window, so the first half is context."""
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def logits_of(model, ids):
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = "cpu"  # an OpenVINO model
    with torch.no_grad():
        out = model(input_ids=ids.to(device), attention_mask=torch.ones_like(ids).to(device))
    return out.logits[0].float().cpu()


device = "cuda"
dtype = torch.bfloat16 if device == "cuda" else torch.float32  # bf16 matmuls crawl on a CPU
tokenizer = AutoTokenizer.from_pretrained(r"<model>")
# .to(), not device_map=: a device map needs accelerate, which not every toolkit's
# environment has, and a 16-bit model moves whole
base = AutoModelForCausalLM.from_pretrained(r"<model>", dtype=dtype).to(device).eval()
candidate = AutoModelForCausalLM.from_pretrained(
    r"<candidate>", dtype=dtype, device_map=device).eval()  # bitsandbytes needs the map

text = open(r"<eval_file>", encoding="utf-8").read()
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
ctx = 512
n = len(tokens) // ctx
n = min(n, 100)

kld = agree = nll_q = nll_b = count = 0.0
for i in range(n):
    ids = tokens[i * ctx:(i + 1) * ctx].unsqueeze(0)
    lb, lq = logits_of(base, ids), logits_of(candidate, ids)
    half = ctx // 2
    pb = torch.log_softmax(lb[half - 1:-1], dim=-1)  # predicts tokens half .. ctx-1
    pq = torch.log_softmax(lq[half - 1:-1], dim=-1)
    target = ids[0, half:]
    kld += float((pb.exp() * (pb - pq)).sum(dim=-1).sum())
    agree += float((pb.argmax(-1) == pq.argmax(-1)).sum())
    nll_q += float(-pq.gather(1, target[:, None]).sum())
    nll_b += float(-pb.gather(1, target[:, None]).sum())
    count += len(target)
    print(f"chunk {i + 1}/{n}: kld {kld / count:.4f} top1 {agree / count:.3f}", flush=True)
result = {
    "kld_mean": kld / count, "top1_agreement": agree / count,
    "ppl": math.exp(nll_q / count), "ppl_base": math.exp(nll_b / count),
    "tokens": int(count), "chunks": n, "ctx": ctx,
}
json.dump(result, open(r"gate.json", "w"), indent=1)
print("gate", json.dumps(result))
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | the 16-bit model the candidate was made from |
| `candidate` | path | required | the quantized checkpoint directory |
| `openvino` | bool | False | load the candidate with Optimum Intel's OVModelForCausalLM (runs on the CPU) |
| `eval_file` | path | required | plain text; the gate scores windows of it |
| `ctx` | int | 512 | tokens per window |
| `chunks` | int | 100 | windows scored; 0 scores the whole file |
| `device` | enum | cuda | where the 16-bit model and a Transformers-loaded candidate run; cpu where the toolkit's PyTorch is a CPU build (Optimum Intel) |
| `gate_file` | path | gate.json | where the numbers are written |

- the same numbers as llama-perplexity --kl-divergence, on the same wikitext-2 windows, so the thresholds in data/quality/gate_thresholds.yaml apply; ppl_base says what the 16-bit model scores on the same tokens
- runs in the environment that can load the candidate: llm-compressor's for FP8 and W4A16 (compressed-tensors), the Transformers toolkit's for NF4 (bitsandbytes), Optimum Intel's for OpenVINO (openvino: true)
- memory: the 16-bit model plus the candidate, plus about 0.3 GB of logits per 512-token window for a 150k vocabulary
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to FORMAT --eval, 2026-09-28): Qwen3-0.6B over 100 windows of wikitext-2, 30-40 s a gate on the GPU (fp8, w4a16, nf4; transformers 5.17.0) and about 140 s on the CPU for OpenVINO (transformers 5.5.4). Its scale is llama.cpp's: on Qwen3-1.7B's Q4_K_M GGUF, loaded dequantized, it scores KLD 0.055 and top-1 0.904 where llama-perplexity gave 0.059 and 0.900

Source: <https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md>
