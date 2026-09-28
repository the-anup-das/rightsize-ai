# whisper.cpp

Generated from `data/recipes/` - do not edit by hand.

Quantizes Whisper models to ggml types and transcribes with them, on CPUs and GPUs

- Runs on: any hardware
- Writes: ggml
- Install: `git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && cmake -B build && cmake --build build -j --config Release` (or brew install whisper-cpp on a Mac)
- Home: <https://github.com/ggml-org/whisper.cpp>

| recipe | stage | families | checked | version |
|---|---|---|---|---|
| `whisper.cpp/quantize` | quantize | audio | docs | v1.9.4 |
| `whisper.cpp/transcribe` | serve | audio | docs | v1.9.4 |

## `whisper.cpp/quantize`

Quantize step; from the documentation; not yet run here at v1.9.4.

Install: git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && cmake -B build && cmake --build build -j --config Release; models: sh ./models/download-ggml-model.sh large-v3-turbo

```bash
./build/bin/whisper-quantize models/ggml-large-v3-turbo.bin models/ggml-large-v3-turbo-q5_0.bin q5_0
```

| input | type | default | notes |
|---|---|---|---|
| `quantize_bin` | path | ./build/bin/whisper-quantize |  |
| `input_model` | path | models/ggml-large-v3-turbo.bin |  |
| `output_model` | path | models/ggml-large-v3-turbo-q5_0.bin |  |
| `qtype` | enum | q5_0 | q5_0, q5_1, q8_0, q4_0, q4_1, q2_k, q3_k, q4_k, q5_k, q6_k |

- the binary is whisper-quantize (the target in examples/quantize/CMakeLists.txt); the README still says ./build/bin/quantize
- rightsize estimate openai/whisper-large-v3-turbo --quant q5_0 --runtime whisper.cpp gives the memory

Source: <https://github.com/ggml-org/whisper.cpp/blob/master/README.md>

## `whisper.cpp/transcribe`

Serve step; from the documentation; not yet run here at v1.9.4.

Install: git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && cmake -B build && cmake --build build -j --config Release, or brew install whisper.cpp

```bash
./build/bin/whisper-cli -m models/ggml-large-v3-turbo-q5_0.bin -l auto -f <audio>
```

| input | type | default | notes |
|---|---|---|---|
| `cli_bin` | path | ./build/bin/whisper-cli |  |
| `model` | path | models/ggml-large-v3-turbo-q5_0.bin |  |
| `audio` | path | required | 16 kHz 16-bit WAV |
| `language` | str | auto |  |

- convert other audio first: ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
- -otxt, -osrt, -ovtt or -oj write a transcript file; ./main is now whisper-cli

Source: <https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/README.md>
