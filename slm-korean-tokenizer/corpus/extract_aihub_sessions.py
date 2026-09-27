#!/usr/bin/env python3
"""Re-extract AI Hub multi-session data as coherent documents for Stage-2.

The Stage-0/1 extractor emitted one line per utterance, which is fine for
tokenizer training but wrong for continued pretraining: it destroys the
conversational context a language model needs. This rebuilds two document-level
streams from the same Training split:

  dialogue    one document per session, speaker turns in order plus the summary
  technical   the retrieved knowledge documents, deduplicated by content

Only `TL_session*` (Training) directories are read. `VL_session*` (Validation)
is the held-out split the evaluation sets are drawn from and is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_record_allowed, assert_source_allowed  # noqa: E402

AIHUB_ROOT = "data/009.전문분야_기술과학_한국어 멀티세션 데이터"
SPEAKER_LABELS = {"speaker1": "질문자", "speaker2": "답변자"}


def clean(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value)).replace("\x00", "")
    return " ".join(text.split()).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def session_documents(payload: dict, path: str) -> tuple[list[dict], list[dict]]:
    """Return (dialogue documents, technical knowledge documents) for one file."""
    dialogues: list[dict] = []
    documents: list[dict] = []
    topic = payload.get("topicInfo") or {}
    categories = " / ".join(
        clean(topic.get(key, "")) for key in ("largeCategory", "mediumCategory", "smallCategory")
        if clean(topic.get(key, ""))
    )
    multisession = clean((payload.get("multisessionInfo") or {}).get("multisessionID", ""))

    for session in payload.get("sessionInfo") or []:
        if not isinstance(session, dict):
            continue
        session_id = clean(session.get("sessionID", "")) or multisession
        turns: list[str] = []
        for dialog in session.get("dialog") or []:
            if not isinstance(dialog, dict):
                continue
            utterance = clean(dialog.get("utterance", ""))
            if utterance:
                label = SPEAKER_LABELS.get(dialog.get("speaker"), "화자")
                turns.append(f"{label}: {utterance}")
            context = dialog.get("context") or {}
            if isinstance(context, dict):
                for content in context.get("contents") or []:
                    if not isinstance(content, dict):
                        continue
                    title = clean(content.get("documentTitle") or "")
                    body_parts = content.get("content")
                    if isinstance(body_parts, str):
                        body_parts = [body_parts]
                    body = clean(" ".join(clean(part) for part in (body_parts or [])
                                          if isinstance(part, str)))
                    text = " ".join(part for part in (title, body) if part).strip()
                    if len(text) >= 100:
                        documents.append({
                            "text": text,
                            "source_id": clean(content.get("documentID") or "") or sha256(text)[:16],
                            "title": title,
                        })

        summary = session.get("sessionSummary") or {}
        summary_text = clean(summary.get("dialogSummary", "")) if isinstance(summary, dict) else ""

        if len(turns) >= 4:
            header = f"[주제: {categories}]" if categories else ""
            body = "\n".join(turns)
            if summary_text:
                body = f"{body}\n\n요약: {summary_text}"
            text = f"{header}\n{body}".strip() if header else body
            dialogues.append({
                "text": text,
                "source_id": session_id or sha256(text)[:16],
                "turns": len(turns),
                "topic": categories,
            })
    return dialogues, documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=AIHUB_ROOT)
    parser.add_argument("--dialogue-out", default="data/processed/stage2_aihub_dialogue.jsonl")
    parser.add_argument("--technical-out", default="data/processed/stage2_aihub_technical.jsonl")
    parser.add_argument("--report", default="outputs/stage2_aihub_extraction.json")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--limit-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    assert_source_allowed("aihub_multisession", root)

    train_dirs = sorted(
        path for path in root.rglob("TL_session*")
        if path.is_dir()
    )
    if not train_dirs:
        raise SystemExit(f"no TL_session* Training directories under {root}")
    files: list[Path] = []
    for directory in train_dirs:
        files.extend(sorted(directory.glob("*.json")))
    if args.limit_files:
        files = files[: args.limit_files]

    counts = Counter()
    dialogue_seen: set[str] = set()
    document_seen: set[str] = set()
    dialogue_path = Path(args.dialogue_out)
    technical_path = Path(args.technical_out)
    dialogue_path.parent.mkdir(parents=True, exist_ok=True)

    with dialogue_path.open("w", encoding="utf-8") as dialogue_handle, \
            technical_path.open("w", encoding="utf-8") as technical_handle:
        for index, path in enumerate(files, 1):
            if index % 5000 == 0:
                print(f"  {index:,}/{len(files):,} files, "
                      f"{counts['dialogue_chars']:,} dialogue chars, "
                      f"{counts['technical_chars']:,} technical chars", flush=True)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                counts["unreadable_files"] += 1
                continue
            counts["files"] += 1
            dialogues, documents = session_documents(payload, str(path))

            for record in dialogues:
                text = record["text"]
                if len(text) < args.min_chars:
                    counts["dialogue_too_short"] += 1
                    continue
                digest = sha256(text)
                if digest in dialogue_seen:
                    counts["dialogue_duplicates"] += 1
                    continue
                dialogue_seen.add(digest)
                assert_record_allowed(text, "aihub_dialogue_session")
                dialogue_handle.write(json.dumps({
                    "text": text, "source": "aihub_dialogue_session",
                    "source_id": record["source_id"], "turns": record["turns"],
                    "topic": record["topic"], "chars": len(text), "sha256": digest,
                }, ensure_ascii=False) + "\n")
                counts["dialogue_documents"] += 1
                counts["dialogue_chars"] += len(text)

            for record in documents:
                text = record["text"]
                digest = sha256(text)
                if digest in document_seen:
                    counts["technical_duplicates"] += 1
                    continue
                document_seen.add(digest)
                assert_record_allowed(text, "aihub_technical_document")
                technical_handle.write(json.dumps({
                    "text": text, "source": "aihub_technical_document",
                    "source_id": record["source_id"], "title": record["title"],
                    "chars": len(text), "sha256": digest,
                }, ensure_ascii=False) + "\n")
                counts["technical_documents"] += 1
                counts["technical_chars"] += len(text)

    report = {
        "root": str(root),
        "split": "Training (TL_session*) only; Validation (VL_session*) untouched",
        "train_directories": [str(path) for path in train_dirs],
        "files_seen": len(files),
        **dict(counts),
        "dialogue_out": args.dialogue_out,
        "technical_out": args.technical_out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
