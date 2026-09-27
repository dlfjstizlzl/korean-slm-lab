"""Fail if the public portfolio Git index contains datasets or model weights."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "slm-korean-tokenizer/data/",
    "slm-korean-tokenizer/outputs/",
)
FORBIDDEN_SUFFIXES = {
    ".arrow", ".bin", ".ckpt", ".csv", ".gguf", ".gz", ".jsonl",
    ".log", ".npz", ".onnx", ".parquet", ".pem", ".pt", ".pth",
    ".safetensors", ".tar", ".tsv", ".zip",
}
MODEL_ALLOWED = {
    "README.md", "MODEL_CARD.md", "MANIFEST.json", "config.json",
    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "merges.txt", "vocab.json", "runtime/system_prompt.txt",
    "runtime/taskplan_v2.gbnf", "adapter/adapter_config.json",
}
REPORT_ALLOWED = {"manager-v1-aggregate.md", "stage2-aggregate.md"}
ROOT_ALLOWED = {
    ".gitignore", "LICENSE", "LICENSE-llama.cpp", "NOTICE", "README.md",
    "THIRD_PARTY_NOTICES.md", "scripts/check_public_release.py",
}
RUNTIME_DOCS = {
    "slm-runtime/CMakeLists.txt", "slm-runtime/README.md",
    "slm-runtime/external/README.md",
}
PYTHON_DOCS = {"slm-korean-tokenizer/requirements.txt"}
SOURCE_SUFFIXES = {".py", ".cpp", ".hpp", ".h"}


def main() -> int:
    paths = [name for name in subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode("utf-8").split("\0") if name]
    errors: list[str] = []
    for name in paths:
        path = Path(name)
        if name.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"data or generated output: {name}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {name}")
        if name.startswith("models/") and path.as_posix()[len("models/"):].split("/", 1)[-1] not in MODEL_ALLOWED:
            errors.append(f"unreviewed model file: {name}")
        if name.startswith("reports/") and path.name not in REPORT_ALLOWED:
            errors.append(f"unreviewed report: {name}")
        if name.startswith("reports/") and (ROOT / path).is_file():
            body = (ROOT / path).read_text(encoding="utf-8")
            if re.search(r"(?im)^.*\bNIKL\b.*(?:\d+\.\d+|tokens/char|NLL)", body):
                errors.append(f"possible NIKL result: {name}")
        allowed = (
            name in ROOT_ALLOWED or name in RUNTIME_DOCS or name in PYTHON_DOCS
            or name.startswith("models/") or name.startswith("reports/")
            or (name.startswith("slm-runtime/") and path.suffix in SOURCE_SUFFIXES)
            or (name.startswith("slm-korean-tokenizer/") and path.suffix in SOURCE_SUFFIXES)
        )
        if not allowed:
            errors.append(f"unreviewed path: {name}")
    if errors:
        print("Public release check FAILED:\n" + "\n".join(errors))
        return 1
    print(f"Public release check passed: {len(paths)} indexed paths; no dataset or weight paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
