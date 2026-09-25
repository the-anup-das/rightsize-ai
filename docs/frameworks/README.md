# Frameworks

Generated from `data/recipes/` - do not edit by hand.

Each framework is a folder in `data/recipes/`: a `framework.yaml` saying what it is for, where it runs, how it installs and how its steps join a plan, and one YAML file per recipe. A plugin package adds a framework the same way; see CONTRIBUTING.md.

| framework | recipes | stages | runs on |
|---|---|---|---|
| [axolotl](axolotl.md) | 1 | finetune | nvidia, amd; on linux |
| [ctranslate2](ctranslate2.md) | 1 | convert | nvidia, cpu |
| [diffusers](diffusers.md) | 1 | quantize | nvidia, amd, intel |
| [llama.cpp](llama.cpp.md) | 6 | calibrate, convert, evaluate, quantize, serve | any hardware; on linux, windows, macos |
| [llm-compressor](llm-compressor.md) | 2 | quantize | nvidia |
| [mlx-lm](mlx-lm.md) | 3 | finetune, quantize, serve | apple; on macos |
| [ollama](ollama.md) | 2 | serve | any hardware; on linux, windows, macos |
| [optimum-intel](optimum-intel.md) | 1 | export | intel, cpu |
| [sentence-transformers](sentence-transformers.md) | 1 | quantize | cpu |
| [transformers](transformers.md) | 1 | quantize | nvidia, amd, intel |
| [trl](trl.md) | 1 | finetune | nvidia, amd, intel |
| [unsloth](unsloth.md) | 1 | finetune | nvidia, amd, intel; on linux, windows; NVIDIA compute capability 7.0+ |
| [vllm](vllm.md) | 1 | serve | nvidia, amd; on linux |
| [whisper.cpp](whisper.cpp.md) | 2 | quantize, serve | any hardware |
