#!/usr/bin/env python3
"""Synthetic Skill-DSL / Manager-plan generator for the Stage-2 corpus.

The Skill DSL is a project-defined format with no natural corpus. Stage 1 showed
the failure mode: 10 templates over one query pool let the model memorize the
skeleton, so Skill-DSL NLL collapsed without any real capability gain.

This generator therefore maximizes structural variety: many plan shapes
(single-step, multi-step, input_from chaining, failure/retry, termination),
several serializations per shape, and query material drawn from real Korean
technical text rather than a fixed list. The builder measures and reports the
resulting skeleton-repetition rate.
"""
from __future__ import annotations

import json
import random

CONTROL = ["<검색>", "<요약>", "<SQL>", "<코드>", "<이전0>", "<이전1>", "<이전2>", "<종료>"]
SKILLS = ["search", "summary", "sql", "code"]
SKILL_TO_TOKEN = {"search": "<검색>", "summary": "<요약>", "sql": "<SQL>", "code": "<코드>"}

TABLES = ["users", "orders", "sessions", "documents", "events", "payments", "devices", "logs"]
COLUMNS = ["id", "name", "created_at", "status", "amount", "user_id", "device_id", "score"]
ERRORS = ["timeout", "not_found", "rate_limited", "invalid_input", "empty_result"]


def _sql(rng: random.Random) -> str:
    table = rng.choice(TABLES)
    columns = ", ".join(rng.sample(COLUMNS, rng.randint(1, 3)))
    shape = rng.randint(0, 4)
    if shape == 0:
        return f"SELECT {columns} FROM {table} WHERE {rng.choice(COLUMNS)} = '{rng.randint(1, 9999)}';"
    if shape == 1:
        return (f"SELECT {rng.choice(COLUMNS)}, COUNT(*) FROM {table} "
                f"GROUP BY {rng.choice(COLUMNS)} ORDER BY COUNT(*) DESC LIMIT {rng.randint(5, 50)};")
    if shape == 2:
        other = rng.choice([t for t in TABLES if t != table])
        return (f"SELECT a.{rng.choice(COLUMNS)}, b.{rng.choice(COLUMNS)} FROM {table} AS a "
                f"JOIN {other} AS b ON a.id = b.{rng.choice(COLUMNS)} "
                f"WHERE a.created_at >= '2026-01-01';")
    if shape == 3:
        return (f"WITH recent AS (SELECT * FROM {table} WHERE created_at >= '2026-06-01') "
                f"SELECT {columns} FROM recent WHERE {rng.choice(COLUMNS)} IS NOT NULL;")
    return f"SELECT COUNT(DISTINCT {rng.choice(COLUMNS)}) FROM {table};"


def _step(rng: random.Random, index: int, query: str, previous: int | None) -> dict:
    skill = rng.choice(SKILLS)
    step: dict = {"step": index, "skill": skill}
    if previous is not None and rng.random() < 0.6:
        step["input_from"] = previous
    else:
        step["input"] = query if skill != "sql" else _sql(rng)
    if skill == "sql" and "input" not in step:
        step["query"] = _sql(rng)
    return step


def _render_plan(rng: random.Random, steps: list[dict], query: str) -> str:
    """One of several serializations, so no single skeleton dominates."""
    style = rng.randint(0, 5)
    if style == 0:
        return json.dumps({"goal": query, "plan": steps}, ensure_ascii=False)
    if style == 1:
        return json.dumps({"goal": query, "plan": steps}, ensure_ascii=False, indent=2)
    if style == 2:
        lines = [f"목표: {query}"]
        for step in steps:
            source = f" <- step{step['input_from']}" if "input_from" in step else ""
            lines.append(f"{step['step']}. {SKILL_TO_TOKEN[step['skill']]} "
                         f"{step.get('input', step.get('query', ''))}{source}")
        lines.append("<종료>")
        return "\n".join(lines)
    if style == 3:
        parts = [SKILL_TO_TOKEN[step["skill"]] + str(step.get("input", step.get("query", "")))
                 for step in steps]
        return "".join(parts) + "<종료>"
    if style == 4:
        lines = [f"사용자: {query}"]
        for step in steps:
            lines.append(f"계획: {SKILL_TO_TOKEN[step['skill']]} "
                         f"{step.get('input', step.get('query', ''))}")
            lines.append(f"결과: step{step['step']} 완료")
        lines.append("응답: 위 결과를 종합해 답변한다. <종료>")
        return "\n".join(lines)
    header = rng.choice(["<이전0>", "<이전1>", "<이전2>"])
    body = json.dumps({"context": header, "goal": query, "plan": steps}, ensure_ascii=False)
    return f"{header}\n{body}"


def _failure_case(rng: random.Random, query: str) -> str:
    skill = rng.choice(SKILLS)
    error = rng.choice(ERRORS)
    style = rng.randint(0, 2)
    attempt = {"skill": skill, "input": query, "status": "error", "error": error}
    retry = {"skill": rng.choice(SKILLS), "input": query, "status": "ok"}
    if style == 0:
        return json.dumps({"goal": query, "attempts": [attempt, retry]}, ensure_ascii=False)
    if style == 1:
        return (f"{SKILL_TO_TOKEN[skill]}{query}\n오류: {error}\n"
                f"재시도: {SKILL_TO_TOKEN[retry['skill']]}{query}\n<종료>")
    return json.dumps({"goal": query, "attempt": attempt, "fallback": retry,
                       "final": "재시도 성공"}, ensure_ascii=False, indent=2)


def generate(queries: list[str], target_chars: int, seed: int = 42) -> list[dict]:
    """Emit Skill/Manager documents until the character budget is met."""
    rng = random.Random(seed)
    pool = list(queries)
    rng.shuffle(pool)
    if not pool:
        return []
    records: list[dict] = []
    total = 0
    index = 0
    while total < target_chars:
        query = pool[index % len(pool)]
        index += 1
        roll = rng.random()
        if roll < 0.20:
            text = _failure_case(rng, query)
            shape = "failure_retry"
        else:
            count = rng.choices([1, 2, 3, 4], weights=[0.30, 0.35, 0.25, 0.10])[0]
            steps: list[dict] = []
            previous = None
            for step_index in range(1, count + 1):
                step = _step(rng, step_index, query, previous)
                steps.append(step)
                previous = step_index
            text = _render_plan(rng, steps, query)
            shape = f"plan_{count}step"
        records.append({
            "text": text, "source": "skill_manager_synthetic",
            "source_id": f"synth-{len(records):07d}", "shape": shape,
            "chars": len(text),
        })
        total += len(text)
    return records


def skeleton(text: str) -> str:
    """Structural fingerprint: which control tokens and shape, ignoring content."""
    marks = [token for token in CONTROL if token in text]
    kind = "json" if text.lstrip().startswith("{") else "text"
    lines = min(text.count("\n"), 8)
    return f"{kind}|{'.'.join(marks)}|{lines}"
