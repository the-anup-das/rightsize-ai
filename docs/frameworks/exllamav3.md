# ExLlamaV3

Generated from `data/recipes/` - do not edit by hand.

Quantizes Hugging Face models to EXL3, a trellis-coded format at any bit rate from 1 to 8 bits, for its own inference library and the servers built on it (TabbyAPI)

- Runs on: nvidia
- Writes: exl3
- Install: `pip install exllamav3` (the PyPI wheel compiles its CUDA kernels on first use and needs nvcc; the release page (github.com/turboderp-org/exllamav3/releases) has prebuilt wheels per CUDA, PyTorch and Python version for Linux and Windows, e.g. exllamav3-1.5.3+cu128.torch2.10.0-cp312-cp312-win_amd64.whl)
- Home: <https://github.com/turboderp-org/exllamav3>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `exllamav3/convert` | quantize | llm | docs | 1.5.3 |

## `exllamav3/convert`

Quantize step; from the documentation; not yet run here at 1.5.3.

Install: pip install exllamav3 (or a prebuilt wheel from the releases page)

```python
"""EXL3: trellis-coded quantization at any bit rate, layer by layer with Hessians computed on
the fly (exllamav3/conversion/convert_model.py, which the repo's convert.py wraps)."""
from huggingface_hub import snapshot_download
from exllamav3.conversion.convert_model import main, parser, prepare

source = snapshot_download(r"<model>")  # a Hub id; a local directory comes back as is
args = parser.parse_args([
    "-i", source, "-o", r"model-exl3", "-w", r"exl3-work",
    "-b", "4.0", "-hb", "6", "-cr", "250", "-cc", "2048",
])
in_args, job_state, ok, err = prepare(args)
if not ok:
    raise SystemExit(f"exllamav3: {err}")
main(in_args, job_state)
```

| input | type | default | notes |
|---|---|---|---|
| `model` | str | required | Hub id or a local directory with config.json, tokenizer.json and safetensors |
| `bits` | float | 4.0 | target average bits per weight; any value from 1 to 8 |
| `head_bits` | int | 6 | bits for the output layer, 1 to 8 |
| `cal_rows` | int | 250 | rows of calibration data |
| `cal_cols` | int | 2048 | tokens per calibration row |
| `output_dir` | path | model-exl3 |  |
| `work_dir` | path | exl3-work | checkpoints and temporary files; needs room for a copy of the output, and -r resumes from it |

- the output loads in ExLlamaV3 itself and the servers built on it (TabbyAPI); vLLM and Transformers do not read EXL3, so there is no --to target and no gate here yet
- -hq raises the bit rate of attention and shared-expert layers for a small size increase; --resume with the work directory continues an interrupted job
- Windows: the PyPI wheel compiles its kernels and needs nvcc and Build Tools; the release page ships prebuilt wheels per CUDA and PyTorch version (cu128 with torch 2.10 and 2.11 at 1.5.3), and triton-windows comes with them. Not run here (2026-09-28)

Source: <https://github.com/turboderp-org/exllamav3/blob/v1.5.3/doc/convert.md>
