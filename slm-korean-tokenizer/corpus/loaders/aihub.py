from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .fineweb import clean_text

SESSION_RE = re.compile(r"(?P<split>[TV]L)_session(?P<session>[234])", re.I)


@dataclass(frozen=True)
class AIHubSource:
    split: str
    session: int
    path: Path
    compressed: bool
    file_count: int
    total_bytes: int


def _session_match(path: Path):
    return SESSION_RE.search(unicodedata.normalize("NFC", str(path)))


def discover_aihub_sources(data_dir: str | Path) -> list[AIHubSource]:
    root = Path(data_dir)
    directories: dict[tuple[str, int], AIHubSource] = {}
    archives: dict[tuple[str, int], AIHubSource] = {}
    for path in root.rglob("*"):
        match = _session_match(path)
        if not match:
            continue
        split = "train" if match.group("split").upper() == "TL" else "valid"
        session = int(match.group("session"))
        key = (split, session)
        if path.is_dir() and path.name.lower() == f"{match.group('split').lower()}_session{session}":
            files = list(path.glob("*.json"))
            if files:
                directories[key] = AIHubSource(split, session, path, False, len(files), sum(p.stat().st_size for p in files))
        elif path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                entries = [item for item in archive.infolist() if item.filename.lower().endswith(".json")]
            archives[key] = AIHubSource(split, session, path, True, len(entries), path.stat().st_size)
    return [directories[key] if key in directories else archives[key] for key in sorted(set(directories) | set(archives))]


def iter_aihub_objects(source: AIHubSource) -> Iterator[tuple[str, dict]]:
    if source.compressed:
        with zipfile.ZipFile(source.path) as archive:
            for name in sorted(item.filename for item in archive.infolist() if item.filename.lower().endswith(".json")):
                with archive.open(name) as handle:
                    yield f"{source.path}!{name}", json.load(handle)
        return
    for path in sorted(source.path.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            yield str(path), json.load(handle)


def _strings(value) -> Iterator[str]:
    if isinstance(value, str):
        if clean_text(value):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def extract_texts(payload: dict) -> tuple[dict[str, list[str]], dict[str, int]]:
    extracted = {"dialogue": [], "documents": [], "queries": [], "summaries": []}
    counts = {"sessions": 0, "utterances": 0, "document_contents": 0, "queries": 0, "summaries": 0}
    sessions = payload.get("sessionInfo") or []
    if not isinstance(sessions, list):
        return extracted, counts
    for session in sessions:
        if not isinstance(session, dict):
            continue
        counts["sessions"] += 1
        dialogs = session.get("dialog") or []
        if isinstance(dialogs, list):
            for dialog in dialogs:
                if not isinstance(dialog, dict):
                    continue
                utterance = dialog.get("utterance")
                if isinstance(utterance, str) and (text := clean_text(utterance)):
                    extracted["dialogue"].append(text)
                    counts["utterances"] += 1
                context = dialog.get("context") or {}
                if isinstance(context, dict):
                    query = context.get("query")
                    if isinstance(query, str) and (text := clean_text(query)):
                        extracted["queries"].append(text)
                        counts["queries"] += 1
                    contents = context.get("contents") or context.get("Contents") or []
                    if isinstance(contents, list):
                        for content in contents:
                            if not isinstance(content, dict):
                                continue
                            title = clean_text(content.get("documentTitle") or "")
                            body = " ".join(clean_text(value) for value in _strings(content.get("content")) if clean_text(value))
                            document = clean_text(" ".join(value for value in (title, body) if value))
                            if document:
                                extracted["documents"].append(document)
                                counts["document_contents"] += 1
                summary = dialog.get("summary")
                if isinstance(summary, str) and (text := clean_text(summary)):
                    extracted["summaries"].append(text)
                    counts["summaries"] += 1
        summary = session.get("sessionSummary") or {}
        if isinstance(summary, dict):
            for key in ("dialogSummary", "apprentice", "wizard"):
                for value in _strings(summary.get(key)):
                    if text := clean_text(value):
                        extracted["summaries"].append(text)
                        counts["summaries"] += 1
    return extracted, counts


def schema_paths(payload, prefix: tuple[str, ...] = ()) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            result.update(schema_paths(value, (*prefix, str(key))))
    elif isinstance(payload, list):
        for value in payload[:5]:
            result.update(schema_paths(value, (*prefix, "[]")))
    else:
        result.add((".".join(prefix), type(payload).__name__))
    return result
