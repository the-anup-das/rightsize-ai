# Optimum Intel

Generated from `data/recipes/` - do not edit by hand.

Exports models to OpenVINO IR with int8 or int4 weights, for Intel CPUs, GPUs and NPUs

- Runs on: intel, cpu
- Writes: openvino
- Install: `pip install -U "optimum-intel[openvino]"` (the openvino extra brings NNCF, which int8 and int4 weight compression need)
- Home: <https://github.com/huggingface/optimum-intel>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `optimum-intel/export-openvino` | export | llm | run | 2.2.0 |

## `optimum-intel/export-openvino`

Export step; run end to end at 2.2.0.

`rightsize quantize MODEL` runs it with `--to openvino-int4` or `--to openvino-int8`.

Install: pip install -U "optimum-intel[openvino]"

```bash
optimum-cli export openvino --model <model> --weight-format int4 --group-size 128 ov_model
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required |  |
| `weight_format` | enum | int4 | int4, int8, fp16, fp32, mxfp4, nf4, cb4 |
| `group_size` | int | 128 | -1 quantizes per column |
| `output_dir` | path | ov_model |  |

- data-aware int4: add --awq --dataset wikitext2 (or --scale-estimation, --gptq); --ratio sets the share of int4 layers, the rest int8
- models over 1B parameters export with int8 weights unless --weight-format says otherwise
- --task is needed for a local model, e.g. text-generation-with-past; flags checked against optimum/commands/export/openvino.py at v2.2.0
- run on Windows with an RTX 4070 Ti SUPER (rightsize quantize --to openvino-int4 and openvino-int8): Qwen3-0.6B, int4 in 43 s (385.1 MB: 196 layers int4_asym in groups of 128, the tied embedding int8 per channel, as NNCF does for embeddings and the last layer) and int8 in 32 s (598 MB); OVModelForCausalLM loads it and answers on the CPU
- gate (--eval) on Qwen3-0.6B: int8 pass (KLD 0.005, top-1 0.956, perplexity 22.12 against 22.13), int4 fail (KLD 0.308, top-1 0.729): data-free int4 of every layer is the worst of the 4-bit results on a model this small; --awq --dataset wikitext2 or a --ratio below 1 are the knobs to try, and 7B-class models are what the int4 default is for

Source: <https://huggingface.co/docs/optimum-intel/en/openvino/export>
