# External dependency

v0.2 was implemented and built against the official `ggml-org/llama.cpp`
checkout at commit:

```text
f172be756ae6e12a6aa294ecaeb2a387649ac6f8
```

This public portfolio export does not bundle the checkout. Prepare it from
the official repository at the pinned commit after cloning:

```bash
git clone https://github.com/ggml-org/llama.cpp.git slm-runtime/external/llama.cpp
git -C slm-runtime/external/llama.cpp checkout f172be756ae6e12a6aa294ecaeb2a387649ac6f8
```

The checkout itself is not part of the public export's Git history.
