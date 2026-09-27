#!/usr/bin/env python3
"""Canonical validator for Manager TaskPlan dataset rows.

Source of truth is slm-runtime (`TaskPlan::from_manager_output` +
`TaskPlan::validate` in src/manager/task_plan.cpp). That code is C++ and
throws on the *first* problem with a free-text message, so it cannot be called
per row from Python or give reason codes. This module therefore mirrors its
acceptance rules check-for-check and adds:

* reason codes for every failure, not only the first
* a stricter `canonical` mode for dataset targets (the exact bytes
  `TaskPlan::to_json()` would emit, no surrounding prose)
* dataset-only diagnostics (UTF-8/U+FFFD, input grounding, redundant tasks)

Parity with the real Runtime is not assumed: `runtime_oracle()` compiles a
thin harness around the unchanged task_plan.cpp and the tests compare both
verdicts on a fuzz corpus and on every generated plan.

Modes
-----
runtime   : accept exactly what the Runtime accepts (prefix/suffix prose is
            tolerated because from_manager_output slices first '{'..last '}')
canonical : runtime-valid AND byte-identical to the canonical serialization
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manager_runtime_registry import (  # noqa: E402
    DEFAULT_RUNTIME_ROOT,
    PROJECT_ROOT,
    RuntimeRegistry,
    load_registry,
)

# --------------------------------------------------------------------------- #
# Reason codes
# --------------------------------------------------------------------------- #

# errors (a plan with any of these is invalid)
NO_JSON_OBJECT = "NO_JSON_OBJECT"
JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
SCHEMA_ERROR = "SCHEMA_ERROR"
UNKNOWN_FIELD = "UNKNOWN_FIELD"
MISSING_ARGUMENT = "MISSING_ARGUMENT"
TYPE_ERROR = "TYPE_ERROR"
EMPTY_PLAN = "EMPTY_PLAN"
TOO_MANY_TASKS = "TOO_MANY_TASKS"
INVALID_TASK_ID = "INVALID_TASK_ID"
INVALID_SKILL = "INVALID_SKILL"
INPUT_SOURCE_CONFLICT = "INPUT_SOURCE_CONFLICT"
EMPTY_INPUT = "EMPTY_INPUT"
INVALID_INPUT_FROM = "INVALID_INPUT_FROM"
SELF_REFERENCE = "SELF_REFERENCE"
FUTURE_REFERENCE = "FUTURE_REFERENCE"
CYCLIC_DEPENDENCY = "CYCLIC_DEPENDENCY"
UTF8_ERROR = "UTF8_ERROR"
NON_CANONICAL = "NON_CANONICAL"
INPUT_NOT_GROUNDED = "INPUT_NOT_GROUNDED"

# warnings (valid, but worth a human look)
DUPLICATE_TASK = "DUPLICATE_TASK"
REDUNDANT_SAME_SKILL_CHAIN = "REDUNDANT_SAME_SKILL_CHAIN"
DISCARDED_OUTPUT = "DISCARDED_OUTPUT"
UNICODE_WHITESPACE_INPUT = "UNICODE_WHITESPACE_INPUT"

ERROR_CODES = (
    NO_JSON_OBJECT, JSON_PARSE_ERROR, SCHEMA_ERROR, UNKNOWN_FIELD, MISSING_ARGUMENT,
    TYPE_ERROR, EMPTY_PLAN, TOO_MANY_TASKS, INVALID_TASK_ID, INVALID_SKILL,
    INPUT_SOURCE_CONFLICT, EMPTY_INPUT, INVALID_INPUT_FROM, SELF_REFERENCE,
    FUTURE_REFERENCE, CYCLIC_DEPENDENCY, UTF8_ERROR, NON_CANONICAL, INPUT_NOT_GROUNDED,
)
WARNING_CODES = (DUPLICATE_TASK, REDUNDANT_SAME_SKILL_CHAIN, DISCARDED_OUTPUT,
                 UNICODE_WHITESPACE_INPUT)

# errors that the C++ Runtime itself would raise (the rest are dataset policy)
RUNTIME_ERROR_CODES = frozenset(ERROR_CODES) - {UTF8_ERROR, NON_CANONICAL, INPUT_NOT_GROUNDED}

TASK_FIELDS = ("id", "skill", "input", "input_from")
INT_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
# task_plan.cpp `blank()` uses std::isspace in the C locale
ASCII_SPACE = frozenset(" \t\n\v\f\r")
JSON_MAX_DEPTH = 32


@dataclass
class Issue:
    code: str
    message: str
    task: int | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "task": self.task}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    plan: dict | None = None

    @property
    def codes(self) -> list[str]:
        return [issue.code for issue in self.errors]

    @property
    def warning_codes(self) -> list[str]:
        return [issue.code for issue in self.warnings]

    @property
    def runtime_valid(self) -> bool:
        return not any(code in RUNTIME_ERROR_CODES for code in self.codes)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


class _ParseFailure(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Parsing (mirrors JsonParser in task_plan.cpp)
# --------------------------------------------------------------------------- #

def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _ParseFailure("duplicate object key")
        result[key] = value
    return result


def _reject_float(text):
    raise _ParseFailure("task plan numbers must be integers")


def _reject_constant(text):
    raise _ParseFailure(f"unexpected value {text}")


def _check_tree(value, depth: int = 0) -> None:
    if depth > JSON_MAX_DEPTH:
        raise _ParseFailure("JSON nesting is too deep")
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise _ParseFailure("lone unicode surrogate")
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        if not -INT64_MAX - 1 <= value <= INT64_MAX:
            raise _ParseFailure("integer is out of range")
    elif isinstance(value, list):
        for item in value:
            _check_tree(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_tree(key, depth + 1)
            _check_tree(item, depth + 1)


def _parse(text: str, strict_bounds: bool):
    """Return the parsed root object or raise _ParseFailure / return None."""
    if strict_bounds:
        body = text
    else:
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last < 0 or last < first:
            return None
        body = text[first:last + 1]
    try:
        value = json.loads(body, object_pairs_hook=_no_duplicates,
                           parse_float=_reject_float, parse_constant=_reject_constant)
    except _ParseFailure:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise _ParseFailure(str(error)) from error
    _check_tree(value)
    return value


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _blank(value: str) -> bool:
    return all(char in ASCII_SPACE for char in value)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def canonical_json(plan: dict) -> str:
    """Byte-identical to TaskPlan::to_json() for plans without control chars."""
    pieces = []
    for task in plan["tasks"]:
        head = '{"id":%d,"skill":%s,' % (task["id"], json.dumps(task["skill"], ensure_ascii=False))
        if "input" in task:
            pieces.append(head + '"input":' + json.dumps(task["input"], ensure_ascii=False) + "}")
        else:
            pieces.append(head + '"input_from":%d}' % task["input_from"])
    return '{"tasks":[' + ",".join(pieces) + "]}"


def _validate_structure(root, registry: RuntimeRegistry, errors: list[Issue]) -> list[dict] | None:
    if not isinstance(root, dict):
        errors.append(Issue(SCHEMA_ERROR, "task plan root must be an object"))
        return None
    if set(root) != {"tasks"}:
        extra = sorted(set(root) - {"tasks"})
        if "tasks" not in root:
            errors.append(Issue(MISSING_ARGUMENT, "task plan is missing 'tasks'"))
        if extra:
            errors.append(Issue(UNKNOWN_FIELD, f"task plan root may contain only 'tasks' (got {extra})"))
        return None
    tasks = root["tasks"]
    if not isinstance(tasks, list):
        errors.append(Issue(TYPE_ERROR, "task plan field 'tasks' has the wrong type"))
        return None

    parsed: list[dict] = []
    structural_ok = True
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(Issue(SCHEMA_ERROR, "each task must be an object", index))
            structural_ok = False
            continue
        for key in task:
            if key not in TASK_FIELDS:
                errors.append(Issue(UNKNOWN_FIELD, f"unknown task field: '{key}'", index))
                structural_ok = False
        for name, check in (("id", _is_int), ("skill", lambda v: isinstance(v, str))):
            if name not in task:
                errors.append(Issue(MISSING_ARGUMENT, f"task plan is missing '{name}'", index))
                structural_ok = False
            elif not check(task[name]):
                errors.append(Issue(TYPE_ERROR, f"task plan field '{name}' has the wrong type", index))
                structural_ok = False
        if _is_int(task.get("id")) and not 0 <= task["id"] <= INT_MAX:
            errors.append(Issue(INVALID_TASK_ID, "task id is out of range", index))
            structural_ok = False
        if "input" in task and not isinstance(task["input"], str):
            errors.append(Issue(TYPE_ERROR, "task input must be a string", index))
            structural_ok = False
        if "input_from" in task:
            value = task["input_from"]
            if not _is_int(value):
                errors.append(Issue(TYPE_ERROR, "task input_from must be a non-negative integer", index))
                structural_ok = False
            elif not 0 <= value <= INT_MAX:
                errors.append(Issue(INVALID_INPUT_FROM, "task input_from must be a non-negative integer", index))
                structural_ok = False
        parsed.append(task)
    return parsed if structural_ok else None


def _validate_semantics(tasks: list[dict], registry: RuntimeRegistry, errors: list[Issue]) -> None:
    if not tasks:
        errors.append(Issue(EMPTY_PLAN, "task plan must contain at least one task"))
        return
    if len(tasks) > registry.max_tasks:
        errors.append(Issue(TOO_MANY_TASKS,
                            f"task plan exceeds the maximum task count ({len(tasks)} > {registry.max_tasks})"))
    count = len(tasks)
    for index, task in enumerate(tasks):
        if task["id"] != index:
            errors.append(Issue(INVALID_TASK_ID, "task ids must be unique and ordered from zero", index))
        if task["skill"] not in registry.skills:
            errors.append(Issue(INVALID_SKILL, f"unsupported skill: '{task['skill']}'", index))
        has_input, has_from = "input" in task, "input_from" in task
        if has_input and has_from:
            errors.append(Issue(INPUT_SOURCE_CONFLICT, "task has both input and input_from", index))
        elif not has_input and not has_from:
            errors.append(Issue(MISSING_ARGUMENT, "task requires one of input or input_from", index))
        if has_input and _blank(task["input"]):
            errors.append(Issue(EMPTY_INPUT, "task input must not be empty", index))
        if has_from:
            source = task["input_from"]
            if source >= count:
                errors.append(Issue(INVALID_INPUT_FROM, f"input_from {source} is outside the plan", index))
            elif source == index:
                errors.append(Issue(SELF_REFERENCE, "task references its own output", index))
            elif source > index:
                errors.append(Issue(FUTURE_REFERENCE, f"input_from {source} is a later task", index))
            # the Runtime compares against the task's *id*; with a bad id the
            # position check above can pass while the Runtime still rejects
            elif source >= task["id"]:
                errors.append(Issue(FUTURE_REFERENCE,
                                    f"input_from {source} is not earlier than id {task['id']}", index))

    # generic cycle check over the single-parent dependency graph
    parent = {i: t.get("input_from") for i, t in enumerate(tasks)
              if _is_int(t.get("input_from")) and 0 <= t["input_from"] < count}
    for start in parent:
        seen = {start}
        node = parent.get(start)
        while node is not None:
            if node in seen:
                errors.append(Issue(CYCLIC_DEPENDENCY, "dependency cycle through input_from", start))
                break
            seen.add(node)
            node = parent.get(node)


def _warnings(tasks: list[dict]) -> list[Issue]:
    warnings = []
    seen: dict[tuple, int] = {}
    consumed = {t["input_from"] for t in tasks if "input_from" in t}
    for index, task in enumerate(tasks):
        source = ("input", task["input"]) if "input" in task else ("from", task.get("input_from"))
        key = (task["skill"],) + source
        if key in seen:
            warnings.append(Issue(DUPLICATE_TASK, f"same skill and input as task {seen[key]}", index))
        seen.setdefault(key, index)
        if "input_from" in task:
            origin = task["input_from"]
            if 0 <= origin < len(tasks) and tasks[origin]["skill"] == task["skill"]:
                warnings.append(Issue(REDUNDANT_SAME_SKILL_CHAIN,
                                      f"{task['skill']} applied to its own output", index))
        if index < len(tasks) - 1 and index not in consumed:
            warnings.append(Issue(DISCARDED_OUTPUT,
                                  "output is neither consumed nor final; the Runtime only returns the last task",
                                  index))
        if "input" in task and not _blank(task["input"]) and not task["input"].strip():
            warnings.append(Issue(UNICODE_WHITESPACE_INPUT, "input is only non-ASCII whitespace", index))
    return warnings


def _utf8_issues(text: str) -> list[Issue]:
    issues = []
    if "�" in text:
        issues.append(Issue(UTF8_ERROR, "contains U+FFFD replacement character"))
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        issues.append(Issue(UTF8_ERROR, "not encodable as UTF-8 (lone surrogate)"))
    return issues


def validate_plan_text(text: str | bytes, registry: RuntimeRegistry | None = None, *,
                       mode: str = "canonical", request: str | None = None) -> ValidationResult:
    """Validate raw Manager output.

    `request`, when given, enables INPUT_NOT_GROUNDED: every literal `input`
    must be a verbatim substring of the user request (candidate dataset
    policy, see 01_manager_contract.md; not a Runtime rule).
    """
    if mode not in {"runtime", "canonical"}:
        raise ValueError(f"unknown mode {mode}")
    registry = registry or load_registry()
    errors: list[Issue] = []

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as error:
            return ValidationResult(False, [Issue(UTF8_ERROR, f"invalid UTF-8: {error}")])

    try:
        root = _parse(text, strict_bounds=False)
    except _ParseFailure as error:
        return ValidationResult(False, [Issue(JSON_PARSE_ERROR, str(error))] + _utf8_issues(text))
    if root is None:
        return ValidationResult(False, [Issue(NO_JSON_OBJECT, "manager output contains no JSON object")])

    tasks = _validate_structure(root, registry, errors)
    if tasks is not None:
        _validate_semantics(tasks, registry, errors)
    errors.extend(_utf8_issues(text))

    plan = root if tasks is not None and not errors else None
    if plan is not None and mode == "canonical" and text != canonical_json(plan):
        errors.append(Issue(NON_CANONICAL, "plan text differs from TaskPlan::to_json() serialization"))
    if tasks is not None and request is not None:
        for index, task in enumerate(tasks):
            if isinstance(task.get("input"), str) and task["input"] not in request:
                errors.append(Issue(INPUT_NOT_GROUNDED, "input is not a verbatim span of the request", index))
    warnings = _warnings(tasks) if tasks is not None and not any(
        issue.code in {SCHEMA_ERROR, TYPE_ERROR, MISSING_ARGUMENT} for issue in errors) else []
    return ValidationResult(not errors, errors, warnings, root if not errors else None)


def validate_plan(plan: dict, registry: RuntimeRegistry | None = None, *,
                  request: str | None = None) -> ValidationResult:
    """Validate a plan object by validating its canonical serialization."""
    try:
        text = canonical_json(plan)
    except (KeyError, TypeError, ValueError):
        text = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    return validate_plan_text(text, registry, mode="canonical", request=request)


def plan_depth(plan: dict) -> int:
    """Longest input_from chain, counted in tasks (single task = 1)."""
    depth: dict[int, int] = {}
    for index, task in enumerate(plan["tasks"]):
        depth[index] = depth.get(task["input_from"], 0) + 1 if "input_from" in task else 1
    return max(depth.values()) if depth else 0


# --------------------------------------------------------------------------- #
# Runtime oracle (the real C++ validator)
# --------------------------------------------------------------------------- #

ORACLE_SOURCE = PROJECT_ROOT / "tools" / "manager_taskplan_oracle.cpp"
ORACLE_DIR = PROJECT_ROOT / "outputs" / "manager_dataset" / "oracle_bin"


def build_runtime_oracle(runtime_root: str | Path = DEFAULT_RUNTIME_ROOT) -> Path | None:
    compiler = shutil.which("clang++") or shutil.which("g++")
    runtime_root = Path(runtime_root).resolve()
    plan_cpp = runtime_root / "src" / "manager" / "task_plan.cpp"
    if compiler is None or not plan_cpp.is_file():
        return None
    digest = hashlib.sha256(ORACLE_SOURCE.read_bytes() + plan_cpp.read_bytes()
                            + (runtime_root / "src/manager/task_plan.hpp").read_bytes()).hexdigest()[:16]
    binary = ORACLE_DIR / f"taskplan_oracle_{digest}"
    if not binary.exists():
        ORACLE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([compiler, "-std=c++20", "-O1", f"-I{runtime_root / 'src'}",
                        str(ORACLE_SOURCE), str(plan_cpp), "-o", str(binary)],
                       check=True, capture_output=True)
    return binary


def runtime_oracle(texts: list[str], runtime_root: str | Path = DEFAULT_RUNTIME_ROOT) -> list[tuple[bool, str]]:
    """Run texts through the unchanged Runtime TaskPlan code. [(accepted, detail)]"""
    binary = build_runtime_oracle(runtime_root)
    if binary is None:
        raise RuntimeError("C++ compiler or runtime sources unavailable")
    payload = "".join(text.encode("utf-8", "surrogatepass").hex() + "\n" for text in texts)
    completed = subprocess.run([str(binary)], input=payload, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    if len(lines) != len(texts):
        raise RuntimeError("oracle output does not match input count")
    return [(line.startswith("OK "), line[3:] if line.startswith("OK ") else line[4:]) for line in lines]


# Fixed Runtime-parity corpus: (raw Manager output, expected primary code or None)
PARITY_CASES: tuple[tuple[str, str | None], ...] = (
    ('{"tasks":[{"id":0,"skill":"search","input":"x"}]}', None),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":1,"skill":"summary","input_from":0}]}', None),
    ('prose {"tasks":[{"id":0,"skill":"summary","input":"x"}]} more', None),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":1,"skill":"summary","input_from":0},'
     '{"id":2,"skill":"search","input_from":0}]}', None),
    ('{"tasks":[{"id":0,"skill":"search","input":"한국어 \\"인용\\" \\u00e9"}]}', None),
    ('{"tasks":[{"id":-0,"skill":"search","input":"x"}]}', None),
    ('{"tasks":[{"id":0,"skill":"search","input":"　"}]}', None),
    ('no json at all', NO_JSON_OBJECT),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"}', JSON_PARSE_ERROR),
    ('only an opening brace {', NO_JSON_OBJECT),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":0.0,"skill":"search","input":"x"}]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":01,"skill":"search","input":"x"}]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input":"x","input":"y"}]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input":"\\ud800"}]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"}]} {}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input":"a\tb"}]}', JSON_PARSE_ERROR),
    ('{"tasks":[{"id":99999999999999999999,"skill":"search","input":"x"}]}', JSON_PARSE_ERROR),
    # the Runtime slices first '{'..last '}', so a bare array becomes its inner object
    ('[{"id":0,"skill":"search","input":"x"}]', MISSING_ARGUMENT),
    ('{"tasks":["search"]}', SCHEMA_ERROR),
    ('{"plan":[{"id":0,"skill":"search","input":"x"}]}', MISSING_ARGUMENT),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"}],"note":"x"}', UNKNOWN_FIELD),
    ('{"tasks":[{"id":0,"skill":"search","input":"x","why":"x"}]}', UNKNOWN_FIELD),
    ('{"tasks":{"id":0}}', TYPE_ERROR),
    ('{"tasks":[{"id":"0","skill":"search","input":"x"}]}', TYPE_ERROR),
    ('{"tasks":[{"id":true,"skill":"search","input":"x"}]}', TYPE_ERROR),
    ('{"tasks":[{"id":0,"skill":["search"],"input":"x"}]}', TYPE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input":null}]}', TYPE_ERROR),
    ('{"tasks":[{"id":0,"skill":"search","input_from":"0"}]}', TYPE_ERROR),
    ('{"tasks":[{"skill":"search","input":"x"}]}', MISSING_ARGUMENT),
    ('{"tasks":[{"id":0,"input":"x"}]}', MISSING_ARGUMENT),
    ('{"tasks":[{"id":0,"skill":"search"}]}', MISSING_ARGUMENT),
    ('{"tasks":[]}', EMPTY_PLAN),
    ('{"tasks":[' + ",".join('{"id":%d,"skill":"search","input":"x"}' % i for i in range(9)) + ']}',
     TOO_MANY_TASKS),
    ('{"tasks":[{"id":1,"skill":"search","input":"x"}]}', INVALID_TASK_ID),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":0,"skill":"summary","input_from":0}]}',
     INVALID_TASK_ID),
    ('{"tasks":[{"id":-1,"skill":"search","input":"x"}]}', INVALID_TASK_ID),
    ('{"tasks":[{"id":0,"skill":"browser","input":"x"}]}', INVALID_SKILL),
    ('{"tasks":[{"id":0,"skill":"sql","input":"x"}]}', INVALID_SKILL),
    ('{"tasks":[{"id":0,"skill":"Search","input":"x"}]}', INVALID_SKILL),
    ('{"tasks":[{"id":0,"skill":"search","input":"x","input_from":0}]}', INPUT_SOURCE_CONFLICT),
    ('{"tasks":[{"id":0,"skill":"search","input":"   "}]}', EMPTY_INPUT),
    ('{"tasks":[{"id":0,"skill":"search","input":""}]}', EMPTY_INPUT),
    ('{"tasks":[{"id":0,"skill":"search","input_from":-1}]}', INVALID_INPUT_FROM),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":1,"skill":"summary","input_from":5}]}',
     INVALID_INPUT_FROM),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":1,"skill":"summary","input_from":1}]}',
     SELF_REFERENCE),
    ('{"tasks":[{"id":0,"skill":"search","input_from":1},{"id":1,"skill":"summary","input_from":0}]}',
     FUTURE_REFERENCE),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"},{"id":1,"skill":"summary","input_from":2},'
     '{"id":2,"skill":"search","input":"y"}]}', FUTURE_REFERENCE),
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="raw Manager output to validate")
    parser.add_argument("--mode", choices=["runtime", "canonical"], default="canonical")
    parser.add_argument("--request")
    arguments = parser.parse_args()
    result = validate_plan_text(arguments.text, mode=arguments.mode, request=arguments.request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
