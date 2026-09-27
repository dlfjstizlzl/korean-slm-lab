"""Skip artifact-verification tests whose local-only inputs are not in the repository.

Corpora, datasets, checkpoints and row-level predictions are kept out of git (see the top-level
README). Tests that verify those artifacts raise FileNotFoundError for a path under data/ or
outputs/ on a fresh clone; they are reported as skipped with the missing path instead of failing.
Any other error still fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = ("data", "outputs")


def _local_missing(error: BaseException) -> Path | None:
    if not isinstance(error, FileNotFoundError) or not error.filename:
        return None
    try:
        relative = Path(error.filename).resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None
    return relative if relative.parts and relative.parts[0] in LOCAL_ONLY else None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when in ("setup", "call") and report.failed and call.excinfo is not None:
        missing = _local_missing(call.excinfo.value)
        if missing is not None:
            report.outcome = "skipped"
            report.longrepr = (str(item.path), item.location[1] or 0,
                               f"Skipped: local artifact not in repository: {missing}")
