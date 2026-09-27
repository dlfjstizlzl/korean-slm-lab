# Qwen3-0.6B-KO32K-Stage2-signal2678 (base)

Metadata and tokenizer for the Stage-2 canonical model (`signal2678`). Model weights
and training state are not distributed in this repository. The manifest records
hashes of the original release files, including the absent weights.

- Architecture: Qwen3ForCausalLM, 28 layers, hidden 1024, float32, tied embeddings
- Tokenizer: KO-32K ByteLevel BPE (vocab 32,000), ChatML special tokens `<|im_start|>`/`<|im_end|>`
- Role: general Korean base; the starting point of Manager-SLM v1. Not instruction-tuned for TaskPlan output
  (raw TaskPlan schema validity on the Manager sealed test: 0.004).

The tokenizer can be loaded from this directory. Model inference requires
reconstructing the weights from the upstream Qwen3-0.6B model using the
training workflow documented in the repository.
