from __future__ import annotations

from pathlib import Path
from typing import Iterator


def discover_nikl_files(data_dir: str | Path) -> list[Path]:
    """Discover future NIKL files without requiring them to exist."""
    root = Path(data_dir)
    candidates = [root / "raw" / "nikl", root / "nikl"]
    return sorted(path for base in candidates if base.exists() for path in base.rglob("*") if path.is_file())


def iter_nikl_documents(data_dir: str | Path) -> Iterator[str]:
    if discover_nikl_files(data_dir):
        raise NotImplementedError("NIKL files were detected, but their schema-specific loader has not been implemented yet")
    return
    yield  # pragma: no cover
