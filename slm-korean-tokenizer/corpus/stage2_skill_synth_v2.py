#!/usr/bin/env python3
"""Skill-DSL / Manager-plan generator, revision 2.

Diagnosis that drove this revision (see `skill_diagnose.py`):

* The query *pool* was not the problem — 50,259 harvested sentences were 100%
  unique, and Wikipedia contributed only 12% of them.
* Stripping control tokens and JSON scaffolding *raised* duplicate-4-gram from
  0.2356 to 0.2913, so the scaffold was diluting repetition, not causing it.
* The real driver was the generator restating one query verbatim in every step
  of a plan. Duplicate-4-gram rose monotonically with step count
  (1-step 0.1766, 2-step 0.1987, 3-step 0.2137, 4-step 0.2424) and peaked on
  failure/retry at 0.3794, which repeats the query in both the attempt and the
  retry.
* A fixed line-plan footer ("응답: 위 결과를 종합해 답변한다. <종료>") appeared
  2,032 times verbatim.

So revision 2 changes what a document *says*, not how the metric is computed:
each step carries its own derived sub-task, later steps reference earlier ones
instead of restating them, footers vary, and query material is drawn from
several positions in a document rather than its first sentence alone.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata

CONTROL = ["<검색>", "<요약>", "<SQL>", "<코드>", "<이전0>", "<이전1>", "<이전2>", "<종료>"]
SKILLS = ["search", "summary", "sql", "code"]
SKILL_TO_TOKEN = {"search": "<검색>", "summary": "<요약>", "sql": "<SQL>", "code": "<코드>"}

TABLES = ["users", "orders", "sessions", "documents", "events", "payments", "devices", "logs",
          "articles", "metrics", "jobs", "reports"]
COLUMNS = ["id", "name", "created_at", "status", "amount", "user_id", "device_id", "score",
           "updated_at", "category", "duration", "source"]
ERRORS = ["timeout", "not_found", "rate_limited", "invalid_input", "empty_result",
          "parse_error", "permission_denied"]

# --------------------------------------------------------------------------- #
# Query material
# --------------------------------------------------------------------------- #

WIKI_MARKUP = re.compile(r"'''|\[\[|\]\]|\{\{|\}\}|<ref[^>]*>|</ref>")
EMPTY_PARENS = re.compile(r"\(\s*[,\s]*\)")
# Stripping wiki markup leaves debris like "(, , math )" where a bolded
# romanization used to be; tidy the remains rather than shipping them.
PAREN_DEBRIS = re.compile(r"\(\s*[,\s]+")
PAREN_TAIL = re.compile(r"[,\s]+\)")
UNBALANCED_PAREN = re.compile(r"\([^)]*$")
CITATION = re.compile(r"\[\d+\]")
BOILERPLATE = {"thumb", "섬네일", "로고", "왼쪽", "오른쪽", "그림", "사진", "표"}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_candidate(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = WIKI_MARKUP.sub(" ", text)
    text = CITATION.sub(" ", text)
    text = PAREN_DEBRIS.sub("(", text)
    text = PAREN_TAIL.sub(")", text)
    text = EMPTY_PARENS.sub(" ", text)
    return " ".join(text.split()).strip(" .,:;·-")


def is_usable_sentence(text: str) -> bool:
    """Reject list headings, citation-only fragments and symbol/number noise."""
    if not 20 <= len(text) <= 140:
        return False
    if text.strip() in BOILERPLATE or text.split()[0] in BOILERPLATE:
        return False
    letters = sum(1 for char in text if char.isalpha())
    if letters / len(text) < 0.55:
        return False
    korean = sum(1 for char in text if "가" <= char <= "힣")
    if korean / len(text) < 0.25:
        return False
    if text.count("*") or text.count("|") > 1:
        return False
    return True


def extract_candidates(text: str, limit: int = 4) -> list[str]:
    """Deterministically take sentences from several positions in a document."""
    sentences = [clean_candidate(piece) for piece in SENTENCE_SPLIT.split(text)]
    usable = [piece for piece in sentences if is_usable_sentence(piece)]
    if not usable:
        return []
    if len(usable) <= limit:
        picks = list(usable)
    else:
        # opening, one-third, two-thirds, and a later sentence
        positions = [0, len(usable) // 3, (2 * len(usable)) // 3, len(usable) - 1]
        seen: set[int] = set()
        picks = []
        for position in positions[:limit]:
            if position not in seen:
                seen.add(position)
                picks.append(usable[position])
    # a two-sentence span adds longer context material
    if len(usable) >= 2:
        span = f"{usable[0]} {usable[1]}"
        if len(span) <= 220:
            picks.append(span)
    return picks


def keywords(text: str, count: int = 3) -> list[str]:
    # Split on brackets too, so a token can never carry an unbalanced "(" into a
    # generated question.
    words = [word.strip(" .,:;·-\"'") for word in re.split(r"[\s()\[\]{}]+", text)]
    words = [word for word in words if len(word) >= 2 and any(ch.isalpha() for ch in word)]
    ordered = sorted(set(words), key=lambda word: (-len(word), word))
    return ordered[:count]


# --------------------------------------------------------------------------- #
# Query modes
# --------------------------------------------------------------------------- #

def make_query(rng: random.Random, candidate: str, mode: str) -> str:
    terms = keywords(candidate)
    head = terms[0] if terms else candidate[:20]
    second = terms[1] if len(terms) > 1 else head
    if mode == "direct":
        return candidate
    if mode == "two_sentence":
        return candidate
    head = head.strip("()[]{} ")
    second = second.strip("()[]{} ")
    if mode == "keyword_question":
        return f"{head}에 대해 무엇을 알 수 있는가?"
    if mode == "entity_lookup":
        return f"{head} 정의와 특징"
    if mode == "summary_request":
        return f"{head} 관련 내용을 요약해 줘"
    if mode == "comparison":
        return f"{head}와 {second}의 차이"
    if mode == "extraction":
        return f"{head}에서 핵심 항목만 뽑아 줘"
    return f"{head} 내용을 표 형태로 정리"


QUERY_MODES = ["direct", "two_sentence", "keyword_question", "entity_lookup",
               "summary_request", "comparison", "extraction", "transformation"]


def sql_statement(rng: random.Random) -> str:
    table = rng.choice(TABLES)
    columns = ", ".join(rng.sample(COLUMNS, rng.randint(1, 3)))
    shape = rng.randint(0, 5)
    if shape == 0:
        return f"SELECT {columns} FROM {table} WHERE {rng.choice(COLUMNS)} = '{rng.randint(1, 9999)}';"
    if shape == 1:
        column = rng.choice(COLUMNS)
        return (f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} "
                f"ORDER BY COUNT(*) DESC LIMIT {rng.randint(5, 50)};")
    if shape == 2:
        other = rng.choice([t for t in TABLES if t != table])
        return (f"SELECT a.{rng.choice(COLUMNS)}, b.{rng.choice(COLUMNS)} FROM {table} AS a "
                f"JOIN {other} AS b ON a.id = b.{rng.choice(COLUMNS)} "
                f"WHERE a.created_at >= '{rng.randint(2024, 2026)}-01-01';")
    if shape == 3:
        return (f"WITH recent AS (SELECT * FROM {table} WHERE created_at >= "
                f"'{rng.randint(2024, 2026)}-06-01') SELECT {columns} FROM recent "
                f"WHERE {rng.choice(COLUMNS)} IS NOT NULL;")
    if shape == 4:
        return f"SELECT COUNT(DISTINCT {rng.choice(COLUMNS)}) FROM {table};"
    return (f"SELECT {columns} FROM {table} ORDER BY {rng.choice(COLUMNS)} "
            f"{rng.choice(['ASC', 'DESC'])} LIMIT {rng.randint(10, 200)};")


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #

def build_steps(rng: random.Random, candidates: list[str], count: int) -> list[dict]:
    """Every step gets its own sub-task; later steps reference, never restate."""
    steps: list[dict] = []
    for index in range(1, count + 1):
        skill = rng.choice(SKILLS)
        step: dict = {"step": index, "skill": skill}
        if index > 1 and rng.random() < 0.75:
            # Reference the previous result rather than repeating the query.
            step["input_from"] = index - 1
            if rng.random() < 0.4:
                step["note"] = rng.choice(
                    ["이전 결과를 활용", "앞 단계 출력 기준", "직전 응답을 입력으로"])
        else:
            candidate = candidates[(index - 1) % len(candidates)]
            mode = QUERY_MODES[rng.randrange(len(QUERY_MODES))]
            if skill == "sql":
                step["query"] = sql_statement(rng)
            else:
                step["input"] = make_query(rng, candidate, mode)
                step["mode"] = mode
        steps.append(step)
    return steps


CLOSINGS = [
    "위 결과를 종합해 답변한다.", "최종 응답을 사용자에게 전달한다.",
    "각 단계 출력을 합쳐 정리한다.", "확인된 내용만 요약해 회신한다.",
    "부족한 부분은 추가 조회한다.", "결과를 표로 정리해 전달한다.",
    "핵심 항목만 추려 보고한다.", "단계별 산출물을 연결해 마무리한다.",
]

OPENINGS = ["목표", "요청", "질의", "작업", "사용자 요청", "처리 대상"]


def render(rng: random.Random, steps: list[dict], goal: str) -> tuple[str, str]:
    style = rng.randrange(7)
    if style == 0:
        return json.dumps({"goal": goal, "plan": steps}, ensure_ascii=False), "json_compact"
    if style == 1:
        return json.dumps({"goal": goal, "plan": steps}, ensure_ascii=False, indent=2), "json_indented"
    if style == 2:
        # key order variant, optional metadata
        payload = {"plan": steps, "goal": goal}
        if rng.random() < 0.5:
            payload["steps"] = len(steps)
        return json.dumps(payload, ensure_ascii=False), "json_reordered"
    if style == 3:
        lines = [f"{rng.choice(OPENINGS)}: {goal}"]
        for step in steps:
            if "input_from" in step:
                body = f"step{step['input_from']} 결과 사용"
            else:
                body = str(step.get("input", step.get("query", "")))
            lines.append(f"{step['step']}. {SKILL_TO_TOKEN[step['skill']]} {body}")
        lines.append(f"{rng.choice(CLOSINGS)} <종료>")
        return "\n".join(lines), "line_plan"
    if style == 4:
        parts = []
        for step in steps:
            body = (f"step{step['input_from']}" if "input_from" in step
                    else str(step.get("input", step.get("query", ""))))
            parts.append(SKILL_TO_TOKEN[step["skill"]] + body)
        return "".join(parts) + "<종료>", "dsl_inline"
    if style == 5:
        header = rng.choice(["<이전0>", "<이전1>", "<이전2>"])
        payload = json.dumps({"context": header, "goal": goal, "plan": steps}, ensure_ascii=False)
        return f"{header}\n{payload}", "context_prefixed"
    lines = [f"사용자: {goal}"]
    for step in steps:
        body = (f"step{step['input_from']} 결과" if "input_from" in step
                else str(step.get("input", step.get("query", ""))))
        lines.append(f"계획: {SKILL_TO_TOKEN[step['skill']]} {body}")
        lines.append(f"결과: step{step['step']} {rng.choice(['완료', '성공', '반환됨', '처리됨'])}")
    lines.append(f"응답: {rng.choice(CLOSINGS)} <종료>")
    return "\n".join(lines), "dialog_plan"


def failure_case(rng: random.Random, candidates: list[str], goal: str) -> tuple[str, str]:
    """Retry uses a *different* formulation, so the query is not restated."""
    skill = rng.choice(SKILLS)
    error = rng.choice(ERRORS)
    alternate = candidates[1 % len(candidates)]
    retry_query = make_query(rng, alternate, QUERY_MODES[rng.randrange(len(QUERY_MODES))])
    retry_skill = rng.choice([s for s in SKILLS if s != skill])
    style = rng.randrange(3)
    attempt = {"skill": skill, "input": goal, "status": "error", "error": error}
    retry = {"skill": retry_skill, "input": retry_query, "status": "ok"}
    if style == 0:
        return json.dumps({"goal": goal, "attempts": [attempt, retry]},
                          ensure_ascii=False), "json_compact"
    if style == 1:
        return (f"{SKILL_TO_TOKEN[skill]}{goal}\n오류: {error}\n"
                f"재시도: {SKILL_TO_TOKEN[retry_skill]}{retry_query}\n"
                f"{rng.choice(CLOSINGS)} <종료>"), "line_plan"
    return json.dumps({"goal": goal, "attempt": attempt, "fallback": retry,
                       "final": rng.choice(["재시도 성공", "대체 경로로 완료", "복구됨"])},
                      ensure_ascii=False, indent=2), "json_indented"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def generate(candidate_groups: list[dict], target_chars: int, seed: int = 42) -> list[dict]:
    """`candidate_groups`: [{"candidates": [...], "source": str, "doc_id": str}, ...]"""
    rng = random.Random(seed)
    groups = [group for group in candidate_groups if group.get("candidates")]
    if not groups:
        return []
    order = list(range(len(groups)))
    rng.shuffle(order)

    records: list[dict] = []
    total = 0
    index = 0
    while total < target_chars:
        group = groups[order[index % len(order)]]
        index += 1
        candidates = group["candidates"]
        goal = make_query(rng, candidates[0], QUERY_MODES[rng.randrange(len(QUERY_MODES))])
        if rng.random() < 0.18:
            text, serialization = failure_case(rng, candidates, goal)
            shape = "failure_retry"
        else:
            count = rng.choices([1, 2, 3, 4], weights=[0.28, 0.34, 0.26, 0.12])[0]
            steps = build_steps(rng, candidates, count)
            text, serialization = render(rng, steps, goal)
            shape = f"plan_{count}step"
        records.append({
            "text": text, "source": "skill_manager_synthetic",
            "source_id": f"synth-{len(records):07d}", "shape": shape,
            "serialization": serialization, "query_source": group.get("source", "unknown"),
            "chars": len(text),
        })
        total += len(text)
    return records


def skeleton(text: str) -> str:
    marks = [token for token in CONTROL if token in text]
    kind = "json" if text.lstrip().startswith("{") else "text"
    lines = min(text.count("\n"), 8)
    return f"{kind}|{'.'.join(marks)}|{lines}"
