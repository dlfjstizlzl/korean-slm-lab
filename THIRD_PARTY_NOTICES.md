# Third-party sources

This export contains source code and aggregate portfolio reports only. It does
not redistribute any training or evaluation dataset, corpus excerpt, row-level
prediction, base-model weight, fine-tuned weight, or GGUF file.

| Source | Use in the research | Terms | Included here |
|---|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Base model and tokenizer configuration | Apache-2.0 | Derived configuration and tokenizer files only; no weights |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ inference library | MIT | Not bundled; build instructions pin the upstream commit |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Tokenizer and model training | ODC-By 1.0 | No data |
| AI Hub technology/science Korean multi-session data | Tokenizer, model and Manager research | AI Hub usage terms | No data or row-level examples |
| National Institute of Korean Language Newspaper Corpus 2025 v1.1 | External evaluation only | Individual usage agreement; provider approval required before public disclosure of results | No data or NIKL-derived results |
| Wikipedia, BigCode, Spider | Structured and mixed-source research | Original source and dataset terms | No data |

`slm-runtime/tools/taskplan_tool.cpp` includes a short UTF-8 decoding function
mirroring llama.cpp's MIT-licensed implementation. The llama.cpp authors retain
the copyright to that excerpt; its license text is included as
`LICENSE-llama.cpp`.
