# Models

Model weights live here and are git-ignored. IndianTaxGPT runs a quantized
[Llama 2 7B Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) model on the CPU through
[ctransformers](https://github.com/marella/ctransformers).

Download the default model:

```bash
python -m indiantaxgpt.download_model
```

| File | Size | RAM needed | Notes |
|------|------|------------|-------|
| `llama-2-7b-chat.Q4_K_M.gguf` | ~4.1 GB | 8 GB | Default. Good balance of speed and quality. |
| `llama-2-7b-chat.Q5_K_M.gguf` | ~4.8 GB | 8–16 GB | Slightly better answers, slower. |
| `llama-2-7b-chat.Q8_0.gguf` | ~7.2 GB | 16 GB | Closest to the full model, slowest. |

To use a different file:

```bash
python -m indiantaxgpt.download_model --file llama-2-7b-chat.Q8_0.gguf
```

Then set `LLM_MODEL_PATH=models/llama-2-7b-chat.Q8_0.gguf` in `.env`.

The first version of this project used GGML `.bin` files. GGUF has since replaced that format in
the llama.cpp ecosystem, so new downloads should use GGUF.
