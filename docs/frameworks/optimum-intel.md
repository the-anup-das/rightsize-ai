# Optimum Intel

Generated from `data/recipes/` - do not edit by hand.

Exports models to OpenVINO IR with int8 or int4 weights, for Intel CPUs, GPUs and NPUs

- Runs on: intel, cpu
- Writes: openvino
- Install: `pip install -U optimum-intel`
- Home: <https://github.com/huggingface/optimum-intel>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `optimum-intel/export-openvino` | export | llm | docs | 2.2.0 |

## `optimum-intel/export-openvino`

Export step; from the documentation; not yet run here at 2.2.0.

Install: pip install -U optimum-intel

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

Source: <https://huggingface.co/docs/optimum-intel/en/openvino/export>
