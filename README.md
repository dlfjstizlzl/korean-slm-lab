# Korean Small Language Model Lab — public code portfolio

This export contains the implementation of a Korean-focused tokenizer research
pipeline, Manager-SLM TaskPlan tooling, and the C++20 runtime. It contains **no
training or evaluation datasets, corpus excerpts, row-level predictions,
generated research outputs, or model weights**. The `models/` directory contains
configuration and tokenizer metadata only. The short reports in `reports/`
contain aggregate results only.

The full research checkout is kept separately. Results derived from the National
Institute of Korean Language's Newspaper Corpus 2025 v1.1 are withheld while
prior approval for public disclosure is pending; no NIKL metrics appear here.

## Components

- `slm-korean-tokenizer/`: corpus builders, tokenizer experiments, training and
  evaluation code, Manager-SLM dataset generation, and tests.
- `slm-runtime/`: model lifetime management, TaskPlan validation and execution,
  mock search/SQL backends, and tests.
- `slm-runtime/external/llama.cpp`: external library obtained separately from
  the official repository at commit `f172be756ae6e12a6aa294ecaeb2a387649ac6f8`.
- `models/`: model cards, configuration, tokenizer files, and original-release
  manifests. The listed weight files are intentionally absent.

## Build and basic test

The C++ runtime currently supports macOS on Apple Silicon. From a clone of
this repository:

```bash
git clone https://github.com/ggml-org/llama.cpp.git slm-runtime/external/llama.cpp
git -C slm-runtime/external/llama.cpp checkout f172be756ae6e12a6aa294ecaeb2a387649ac6f8
cmake -S slm-runtime -B slm-runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build slm-runtime/build -j4
ctest --test-dir slm-runtime/build --output-on-failure
```

For the Python code, use Python 3.11:

```bash
cd slm-korean-tokenizer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_common.py
```

The basic test above runs without private data. The full test suite and model
inference require datasets, generated artifacts, or weights excluded from this
export. No script downloads a
restricted dataset automatically. Obtain any data directly from its provider
under your own terms before using the research workflows.

## Model and data boundary

The research used [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as its
base model. Its weights and all derived weights are absent here. The included
tokenizer metadata is derived from Qwen3-0.6B and training corpora. Model training
and evaluation require external inputs; see the source code and
`THIRD_PARTY_NOTICES.md` for provenance. The included reports do not reproduce
dataset records or model generations.

## License

Repository-owned code and documentation are Apache-2.0 (`LICENSE`, `NOTICE`).
The pinned llama.cpp dependency is MIT-licensed. Source-data and base-model
terms remain those of their original providers; see `THIRD_PARTY_NOTICES.md`.
