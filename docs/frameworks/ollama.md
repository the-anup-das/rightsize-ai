# Ollama

Generated from `data/recipes/` - do not edit by hand.

Serves a GGUF locally from a Modelfile, with the model's chat template and parameters

- Runs on: any hardware; on linux, windows, macos
- Install: `curl -fsSL https://ollama.com/install.sh | sh` (on Windows: irm https://ollama.com/install.ps1 | iex)
- Home: <https://ollama.com>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `ollama/create` | serve | llm | docs | v0.34.4 |
| `ollama/modelfile` | serve | llm | docs | v0.34.4 |

## `ollama/create`

Serve step; from the documentation; not yet run here at v0.34.4.

Install: curl -fsSL https://ollama.com/install.sh | sh (Linux, macOS); irm https://ollama.com/install.ps1 | iex (Windows)

```bash
ollama create <name> -f Modelfile
```

| input | type | default | notes |
|---|---|---|---|
| `name` | str | required | the model name to run it by |
| `modelfile` | path | Modelfile |  |

- then: ollama run NAME

Source: <https://docs.ollama.com/import>

## `ollama/modelfile`

Serve step; from the documentation; not yet run here at v0.34.4.

Install: curl -fsSL https://ollama.com/install.sh | sh (Linux, macOS); irm https://ollama.com/install.ps1 | iex (Windows)

```modelfile
FROM <model_gguf>
PARAMETER num_ctx 8192
```

| input | type | default | notes |
|---|---|---|---|
| `model_gguf` | path | required | the GGUF to import, absolute or relative to the Modelfile |
| `num_ctx` | int | 8192 | context length the plan was sized for |

- save as Modelfile, then run the ollama/create step
- Ollama imports a GGUF as is; ollama create -q quantizes safetensors only
- chat models may need a TEMPLATE block (Go template syntax); check what Ollama detected with ollama show --modelfile NAME
- set num_ctx explicitly: the docs give 2048 on the Modelfile page and 4096 in the FAQ

Source: <https://docs.ollama.com/import>
