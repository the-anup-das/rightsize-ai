# GGUF header fixtures

The first bytes of three files from llama.cpp's CI test models, up to where the tensor data
starts: the header only, no weights. They come from
<https://huggingface.co/ggml-org/models-moved/tree/main/tinyllamas>, llama.cpp conversions of
Andrej Karpathy's TinyStories models (<https://huggingface.co/karpathy/tinyllamas>, MIT).

| File | What it tests |
|---|---|
| `stories260K.head` | a whole header: 19 metadata keys, a 512-token vocabulary, 48 tensors |
| `stories260K-be.head` | the same model written big-endian |
| `stories15M-q8_0-00002-of-00003.head` | the second part of a split model: its own tensors, no model metadata |

Refetch with, for example:

    curl -L -r 0-14175 -o stories260K.head \
      https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories260K.gguf
