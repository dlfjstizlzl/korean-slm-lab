#!/usr/bin/env python3
"""TaskPlan v2 validator (Python mirror of slm-runtime task_plan.cpp).

Runtime behaviour is defined by `TaskPlan::from_manager_output` +
`TaskPlan::validate` in C++. This module repeats the same checks in the same
order and reports the same first error code, so a dataset row can be validated
without a subprocess per row. Parity is enforced, not assumed:
`cpp_validate()` and `cpp_grammar()` run the Runtime's own code through
`slm_taskplan_tool`, and the tests and analysis require zero disagreements.

Dataset-only checks come on top of Runtime validity:

* NON_CANONICAL  - text is not byte-identical to `TaskPlan::to_json()`
* PAYLOAD_COPY   - an instruction repeats >= PAYLOAD_COPY_CHARS consecutive
                   characters of the user's payload (long input must be
                   referenced with the "user" source, not copied)
* USER_SOURCE_WITHOUT_PAYLOAD - "user" source although the request carries no
                   user content
Warnings: UNUSED_TASK, DUPLICATE_TASK, SAME_SKILL_CHAIN.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manager_v2_registry import ManagerContract, ensure_tool, load_contract  # noqa: E402

ROOT_FIELDS = ("status", "message", "tasks", "outputs")
TASK_FIELDS = ("id", "skill", "instruction", "sources")
STATUSES = ("ok", "clarify", "unsupported")
INT_MAX = 2**31 - 1
INT64 = 2**63
JSON_MAX_DEPTH = 32
PAYLOAD_COPY_CHARS = 20
SPACE_CODEPOINTS = frozenset([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680, *range(0x2000, 0x200B),
                              0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF])
DEPENDENCY_CODES = frozenset({"INVALID_SOURCE", "SELF_REFERENCE", "FUTURE_REFERENCE", "DUPLICATE_SOURCE",
                              "SOURCE_COUNT"})
OUTPUT_CODES = frozenset({"EMPTY_OUTPUTS", "INVALID_OUTPUT", "DUPLICATE_OUTPUT"})


class PlanError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class Result:
    valid: bool
    code: str | None = None
    message: str | None = None
    plan: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "code": self.code, "message": self.message, "warnings": self.warnings}


def plan_text_is_invalid(value: str) -> bool:
    return any(ord(c) < 0x20 or ord(c) == 0x7F or 0x80 <= ord(c) <= 0x9F or ord(c) == 0xFFFD for c in value)


def plan_text_is_blank(value: str) -> bool:
    return all(ord(c) in SPACE_CODEPOINTS for c in value)


# --------------------------------------------------------------------------- #
# parse (mirrors JsonParser + from_manager_output)
# --------------------------------------------------------------------------- #

def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanError("JSON_PARSE_ERROR", "duplicate object key")
        result[key] = value
    return result


def _reject_float(_):
    raise PlanError("JSON_PARSE_ERROR", "task plan numbers must be integers")


def _reject_constant(value):
    raise PlanError("JSON_PARSE_ERROR", f"unexpected value {value}")


def _check_tree(value, depth=1):
    # C++ counts the root container as depth 1 and fails beyond 32.
    if isinstance(value, (list, dict)) and depth > JSON_MAX_DEPTH + 1:
        raise PlanError("JSON_PARSE_ERROR", "JSON nesting is too deep")
    if isinstance(value, str):
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise PlanError("JSON_PARSE_ERROR", "unpaired unicode surrogate")
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        if not -INT64 <= value < INT64:
            raise PlanError("JSON_PARSE_ERROR", "integer is out of range")
    elif isinstance(value, list):
        for item in value:
            _check_tree(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_tree(key, depth + 1)
            _check_tree(item, depth + 1)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required(obj: dict, name: str, kind):
    if name not in obj:
        raise PlanError("MISSING_FIELD", f"task plan is missing '{name}'")
    value = obj[name]
    ok = _is_int(value) if kind is int else isinstance(value, kind)
    if not ok:
        raise PlanError("TYPE_ERROR", f"task plan field '{name}' has the wrong type")
    return value


def parse(text: str | bytes) -> dict:
    """Parse raw Manager output into a normalized plan dict or raise PlanError."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PlanError("INVALID_TEXT", f"manager output is not valid UTF-8 ({error})") from error
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PlanError("INVALID_TEXT", "manager output is not valid UTF-8") from error
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last < 0 or last < first:
        raise PlanError("NO_JSON_OBJECT", "manager output contains no JSON object")
    try:
        root = json.loads(text[first:last + 1], object_pairs_hook=_no_duplicates,
                          parse_float=_reject_float, parse_constant=_reject_constant)
    except PlanError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise PlanError("JSON_PARSE_ERROR", str(error)) from error
    _check_tree(root)
    if not isinstance(root, dict):
        raise PlanError("SCHEMA_ERROR", "task plan root must be an object")
    for key in sorted(root):
        if key not in ROOT_FIELDS:
            raise PlanError("UNKNOWN_FIELD", f"unknown task plan field: '{key}'")
    status = _required(root, "status", str)
    if status not in STATUSES:
        raise PlanError("INVALID_STATUS", f"unknown status: '{status}'")
    tasks = _required(root, "tasks", list)
    outputs = _required(root, "outputs", list)
    plan = {"status": status}
    if "message" in root:
        if not isinstance(root["message"], str):
            raise PlanError("TYPE_ERROR", "task plan field 'message' has the wrong type")
        plan["message"] = root["message"]
    parsed_tasks = []
    for item in tasks:
        if not isinstance(item, dict):
            raise PlanError("SCHEMA_ERROR", "each task must be an object")
        for key in sorted(item):
            if key not in TASK_FIELDS:
                raise PlanError("UNKNOWN_FIELD", f"unknown task field: '{key}'")
        task_id = _required(item, "id", int)
        if not 0 <= task_id <= INT_MAX:
            raise PlanError("INVALID_TASK_ID", "task id is out of range")
        skill = _required(item, "skill", str)
        instruction = _required(item, "instruction", str)
        sources = _required(item, "sources", list)
        parsed_sources = []
        for source in sources:
            if isinstance(source, str):
                if source != "user":
                    raise PlanError("INVALID_SOURCE", f"string source must be \"user\", got '{source}'")
                parsed_sources.append("user")
            elif _is_int(source):
                if not 0 <= source <= INT_MAX:
                    raise PlanError("INVALID_SOURCE", "task source id is out of range")
                parsed_sources.append(source)
            else:
                raise PlanError("TYPE_ERROR", "a source must be \"user\" or an integer task id")
        parsed_tasks.append({"id": task_id, "skill": skill, "instruction": instruction,
                             "sources": parsed_sources})
    parsed_outputs = []
    for output in outputs:
        if not _is_int(output):
            raise PlanError("TYPE_ERROR", "outputs must contain integer task ids")
        if not 0 <= output <= INT_MAX:
            raise PlanError("INVALID_OUTPUT", "output task id is out of range")
        parsed_outputs.append(output)
    plan["tasks"] = parsed_tasks
    plan["outputs"] = parsed_outputs
    return plan


# --------------------------------------------------------------------------- #
# validate (mirrors TaskPlan::validate)
# --------------------------------------------------------------------------- #

def validate_semantics(plan: dict, contract: ManagerContract) -> None:
    status, tasks, outputs = plan["status"], plan["tasks"], plan["outputs"]
    if status != "ok":
        if tasks or outputs:
            raise PlanError("STATUS_SEMANTICS", f"{status} plan must have empty tasks and outputs")
        if "message" not in plan:
            raise PlanError("STATUS_SEMANTICS", f"{status} plan requires a message")
        if plan_text_is_invalid(plan["message"]):
            raise PlanError("INVALID_TEXT", "message contains invalid text")
        if plan_text_is_blank(plan["message"]):
            raise PlanError("EMPTY_MESSAGE", "message must not be empty")
        return
    if "message" in plan:
        raise PlanError("STATUS_SEMANTICS", "ok plan must not carry a message")
    if not tasks:
        raise PlanError("EMPTY_PLAN", "ok plan must contain at least one task")
    if not outputs:
        raise PlanError("EMPTY_OUTPUTS", "ok plan must name at least one output")
    if len(tasks) > contract.max_tasks:
        raise PlanError("TOO_MANY_TASKS", "task plan exceeds the maximum task count")
    for index, task in enumerate(tasks):
        if task["id"] != index:
            raise PlanError("INVALID_TASK_ID", "task ids must be unique and ordered from zero")
        spec = contract.skill(task["skill"])
        if spec is None:
            raise PlanError("INVALID_SKILL", f"unsupported skill: '{task['skill']}'")
        if plan_text_is_invalid(task["instruction"]):
            raise PlanError("INVALID_TEXT", "instruction contains invalid text")
        if plan_text_is_blank(task["instruction"]):
            raise PlanError("EMPTY_INSTRUCTION", "task instruction must not be empty")
        seen = []
        for source in task["sources"]:
            if source != "user":
                if source == task["id"]:
                    raise PlanError("SELF_REFERENCE", "task references its own output")
                if source > task["id"]:
                    raise PlanError("FUTURE_REFERENCE", "source must reference an earlier task")
            if source in seen:
                raise PlanError("DUPLICATE_SOURCE", "duplicate task source")
            seen.append(source)
        if not spec.min_sources <= len(task["sources"]) <= spec.max_sources:
            raise PlanError("SOURCE_COUNT", f"{task['skill']} requires {spec.min_sources}.."
                                            f"{spec.max_sources} sources")
    seen_outputs = []
    for output in outputs:
        if output >= len(tasks):
            raise PlanError("INVALID_OUTPUT", "output references a missing task")
        if output in seen_outputs:
            raise PlanError("DUPLICATE_OUTPUT", "duplicate output task id")
        seen_outputs.append(output)


def canonical_json(plan: dict) -> str:
    """Byte-identical to TaskPlan::to_json() (instructions/messages carry no control chars)."""
    dump = lambda value: json.dumps(value, ensure_ascii=False)  # noqa: E731
    head = '{"status":' + dump(plan["status"])
    if plan["status"] != "ok":
        head += ',"message":' + dump(plan["message"])
    tasks = ",".join('{"id":%d,"skill":%s,"instruction":%s,"sources":[%s]}' % (
        task["id"], dump(task["skill"]), dump(task["instruction"]),
        ",".join('"user"' if s == "user" else str(s) for s in task["sources"])) for task in plan["tasks"])
    return head + ',"tasks":[' + tasks + '],"outputs":[' + ",".join(str(o) for o in plan["outputs"]) + "]}"


def warnings_for(plan: dict) -> list[str]:
    warnings = []
    tasks = plan["tasks"]
    consumed = {s for t in tasks for s in t["sources"] if s != "user"}
    seen = set()
    for task in tasks:
        key = (task["skill"], task["instruction"], tuple(map(str, task["sources"])))
        if key in seen:
            warnings.append("DUPLICATE_TASK")
        seen.add(key)
        if task["id"] not in consumed and task["id"] not in plan["outputs"]:
            warnings.append("UNUSED_TASK")
        if any(s != "user" and tasks[s]["skill"] == task["skill"] and task["skill"] != "search"
               for s in task["sources"]):
            warnings.append("SAME_SKILL_CHAIN")
    return warnings


def longest_common_substring(a: str, b: str) -> int:
    if not a or not b:
        return 0
    best = 0
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0] * (len(b) + 1)
        for j, char_b in enumerate(b, 1):
            if char_a == char_b:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def validate(text: str | bytes, contract: ManagerContract | None = None, *, mode: str = "canonical",
             payloads: list[str] | None = None, names: list[str] | None = None) -> Result:
    """mode='runtime' mirrors the Runtime; 'canonical' adds dataset checks.

    `names` are target names (e.g. document titles) that an instruction may
    legitimately repeat; they are removed before the PAYLOAD_COPY comparison.
    """
    contract = contract or load_contract()
    try:
        plan = parse(text)
        validate_semantics(plan, contract)
    except PlanError as error:
        return Result(False, error.code, str(error))
    if mode == "runtime":
        return Result(True, plan=plan, warnings=warnings_for(plan))
    if text != canonical_json(plan):
        return Result(False, "NON_CANONICAL", "plan text differs from TaskPlan::to_json()", plan)
    uses_user = any("user" in task["sources"] for task in plan["tasks"])
    if payloads is not None:
        if uses_user and not any(payloads):
            return Result(False, "USER_SOURCE_WITHOUT_PAYLOAD", "user source but no user content", plan)
        for task in plan["tasks"]:
            instruction = task["instruction"]
            for name in sorted(names or [], key=len, reverse=True):
                instruction = instruction.replace(name, " ")
            for payload in payloads:
                if longest_common_substring(instruction, payload) >= PAYLOAD_COPY_CHARS:
                    return Result(False, "PAYLOAD_COPY", "instruction copies user payload text", plan)
    return Result(True, plan=plan, warnings=warnings_for(plan))


# --------------------------------------------------------------------------- #
# Runtime oracle
# --------------------------------------------------------------------------- #

def _run_tool(mode: str, texts: list[str | bytes]) -> list[str]:
    tool = ensure_tool()
    lines = "".join((t if isinstance(t, bytes) else t.encode("utf-8", "surrogatepass")).hex() + "\n"
                    for t in texts)
    out = subprocess.run([str(tool), mode], input=lines, capture_output=True, text=True, check=True).stdout
    result = out.splitlines()
    if len(result) != len(texts):
        raise RuntimeError(f"{mode}: expected {len(texts)} lines, got {len(result)}")
    return result


def cpp_validate(texts: list[str | bytes]) -> list[tuple[bool, str, str]]:
    """[(accepted, code_or_OK, to_json_or_message)] from the Runtime's C++ code."""
    result = []
    for line in _run_tool("--validate", texts):
        if line.startswith("OK "):
            result.append((True, "OK", line[3:]))
        else:
            _, code, message = (line.split(" ", 2) + [""])[:3]
            result.append((False, code, message))
    return result


def cpp_grammar(texts: list[str | bytes]) -> list[bool]:
    """llama.cpp grammar engine verdict for ManagerClient::json_grammar()."""
    return [line == "ACCEPT" for line in _run_tool("--grammar", texts)]


# --------------------------------------------------------------------------- #
# Parity corpus: (raw output, expected first code or "OK")
# --------------------------------------------------------------------------- #

def _task(i, skill, sources="[]", instruction="지시"):
    return '{"id":%d,"skill":"%s","instruction":"%s","sources":%s}' % (i, skill, instruction, sources)


def _ok(tasks, outputs="[0]"):
    return '{"status":"ok","tasks":[' + ",".join(tasks) + '],"outputs":' + outputs + "}"


SUMMARY = _task(0, "summary", '["user"]')
PARITY_CASES: tuple[tuple[str | bytes, str], ...] = (
    (_ok([_task(0, "search")]), "OK"),
    (_ok([_task(0, "search"), _task(1, "summary", "[0]")], "[1]"), "OK"),
    (_ok([_task(0, "search"), _task(1, "search"), _task(2, "compare", "[0,1]")], "[2]"), "OK"),
    (_ok([_task(0, "compare", '["user"]')]), "OK"),
    (_ok([SUMMARY, _task(1, "extract", '["user"]')], "[0,1]"), "OK"),
    (_ok([_task(0, "sql"), _task(1, "search"), _task(2, "compare", '["user",0,1]')], "[2]"), "OK"),
    ('{"status":"clarify","message":"요약할 내용을 제공해 주세요.","tasks":[],"outputs":[]}', "OK"),
    ('{"status":"unsupported","message":"불가","tasks":[],"outputs":[]}', "OK"),
    ('prose { "status" : "ok" , "tasks" : [ ' + SUMMARY + ' ] , "outputs" : [ 0 ] } more', "OK"),
    (_ok([_task(0, "sql", "[]", '따옴표 \\"x\\" 역슬래시 \\\\ 슬래시 \\/')]), "OK"),
    (_ok([_task(0, "search", "[]", "\\u00e9 escape")]), "OK"),
    ("no json", "NO_JSON_OBJECT"),
    ("[1,2]", "NO_JSON_OBJECT"),
    ("{", "NO_JSON_OBJECT"),
    (_ok([SUMMARY]) + "{}", "JSON_PARSE_ERROR"),
    (_ok([SUMMARY])[:-1] + ",}", "JSON_PARSE_ERROR"),
    ('{"status":"ok","status":"ok","tasks":[],"outputs":[]}', "JSON_PARSE_ERROR"),
    (_ok(['{"id":0.0,"skill":"search","instruction":"x","sources":[]}']), "JSON_PARSE_ERROR"),
    (_ok(['{"id":1e0,"skill":"search","instruction":"x","sources":[]}']), "JSON_PARSE_ERROR"),
    (_ok(['{"id":01,"skill":"search","instruction":"x","sources":[]}']), "JSON_PARSE_ERROR"),
    (_ok(['{"id":99999999999999999999,"skill":"search","instruction":"x","sources":[]}']), "JSON_PARSE_ERROR"),
    (_ok([_task(0, "search", "[]", "\\ud800")]), "JSON_PARSE_ERROR"),
    (_ok([_task(0, "search", "[]", "a\tb")]), "JSON_PARSE_ERROR"),
    ('{"status":"ok",\x0b"tasks":[' + SUMMARY + '],"outputs":[0]}', "JSON_PARSE_ERROR"),
    (_ok([_task(0, "search", "[]", "NaN")]).replace('"NaN"', "NaN"), "JSON_PARSE_ERROR"),
    ('{"status":"ok","tasks":[' + SUMMARY + '],"outputs":[0],"note":1}', "UNKNOWN_FIELD"),
    (_ok(['{"id":0,"skill":"search","instruction":"x","sources":[],"why":1}']), "UNKNOWN_FIELD"),
    ('{"status":"ok","tasks":[{"id":0,"skill":"search","input":"x"}],"outputs":[0]}', "UNKNOWN_FIELD"),
    ('{"tasks":[{"id":0,"skill":"search","input":"x"}]}', "MISSING_FIELD"),
    ('{"status":"ok","outputs":[]}', "MISSING_FIELD"),
    ('{"status":"ok","tasks":[]}', "MISSING_FIELD"),
    (_ok(['{"skill":"search","instruction":"x","sources":[]}']), "MISSING_FIELD"),
    (_ok(['{"id":0,"instruction":"x","sources":[]}']), "MISSING_FIELD"),
    (_ok(['{"id":0,"skill":"search","sources":[]}']), "MISSING_FIELD"),
    (_ok(['{"id":0,"skill":"search","instruction":"x"}']), "MISSING_FIELD"),
    ('{"status":1,"tasks":[],"outputs":[]}', "TYPE_ERROR"),
    ('{"status":"ok","tasks":{},"outputs":[]}', "TYPE_ERROR"),
    ('{"status":"ok","tasks":[],"outputs":0}', "TYPE_ERROR"),
    ('{"status":"clarify","message":null,"tasks":[],"outputs":[]}', "TYPE_ERROR"),
    (_ok(['{"id":"0","skill":"search","instruction":"x","sources":[]}']), "TYPE_ERROR"),
    (_ok(['{"id":true,"skill":"search","instruction":"x","sources":[]}']), "TYPE_ERROR"),
    (_ok(['{"id":0,"skill":["search"],"instruction":"x","sources":[]}']), "TYPE_ERROR"),
    (_ok(['{"id":0,"skill":"search","instruction":1,"sources":[]}']), "TYPE_ERROR"),
    (_ok(['{"id":0,"skill":"search","instruction":"x","sources":{}}']), "TYPE_ERROR"),
    (_ok([_task(0, "summary", "[null]")]), "TYPE_ERROR"),
    (_ok([_task(0, "summary", "[true]")]), "TYPE_ERROR"),
    (_ok([SUMMARY], '["0"]'), "TYPE_ERROR"),
    ('{"status":"ok","tasks":["x"],"outputs":[0]}', "SCHEMA_ERROR"),
    ('{"status":"done","tasks":[],"outputs":[]}', "INVALID_STATUS"),
    ('{"status":"OK","tasks":[],"outputs":[]}', "INVALID_STATUS"),
    ('{"status":"ok","message":"x","tasks":[' + SUMMARY + '],"outputs":[0]}', "STATUS_SEMANTICS"),
    ('{"status":"clarify","tasks":[],"outputs":[]}', "STATUS_SEMANTICS"),
    ('{"status":"unsupported","message":"x","tasks":[' + SUMMARY + '],"outputs":[]}', "STATUS_SEMANTICS"),
    ('{"status":"clarify","message":"x","tasks":[],"outputs":[0]}', "STATUS_SEMANTICS"),
    ('{"status":"clarify","message":"","tasks":[],"outputs":[]}', "EMPTY_MESSAGE"),
    ('{"status":"clarify","message":" 　","tasks":[],"outputs":[]}', "EMPTY_MESSAGE"),
    ('{"status":"clarify","message":"\\n","tasks":[],"outputs":[]}', "INVALID_TEXT"),
    ('{"status":"ok","tasks":[],"outputs":[]}', "EMPTY_PLAN"),
    (_ok([SUMMARY], "[]"), "EMPTY_OUTPUTS"),
    (_ok([_task(i, "search") for i in range(9)], "[8]"), "TOO_MANY_TASKS"),
    (_ok([_task(1, "search")], "[0]"), "INVALID_TASK_ID"),
    (_ok([_task(0, "search"), _task(0, "search")], "[0]"), "INVALID_TASK_ID"),
    (_ok(['{"id":-1,"skill":"search","instruction":"x","sources":[]}']), "INVALID_TASK_ID"),
    (_ok([_task(0, "translate", '["user"]')]), "INVALID_SKILL"),
    (_ok([_task(0, "code")]), "INVALID_SKILL"),
    (_ok([_task(0, "Search")]), "INVALID_SKILL"),
    (_ok([_task(0, "search", "[]", "")]), "EMPTY_INSTRUCTION"),
    (_ok([_task(0, "search", "[]", "   ")]), "EMPTY_INSTRUCTION"),
    (_ok([_task(0, "search", "[]", "　 ")]), "EMPTY_INSTRUCTION"),
    (_ok([_task(0, "search", "[]", "깨짐�")]), "INVALID_TEXT"),
    (_ok([_task(0, "search", "[]", "a\\u0007b")]), "INVALID_TEXT"),
    (_ok([_task(0, "search", "[]", "c1\\u0085")]), "INVALID_TEXT"),
    (_ok([_task(0, "search", "[]", "x")]).encode("utf-8").replace(b'"x"', b'"\xc3\x28"'), "INVALID_TEXT"),
    (_ok([_task(0, "search", "[]", "x")]).encode("utf-8").replace(b'"x"', b'"\xed\xa0\x80"'), "INVALID_TEXT"),
    (_ok([_task(0, "summary", '["users"]')]), "INVALID_SOURCE"),
    (_ok([_task(0, "summary", "[-1]")]), "INVALID_SOURCE"),
    (_ok([_task(0, "summary", "[4294967296]")]), "INVALID_SOURCE"),
    (_ok([_task(0, "search"), _task(1, "summary", "[1]")], "[1]"), "SELF_REFERENCE"),
    (_ok([_task(0, "summary", "[1]"), _task(1, "search")]), "FUTURE_REFERENCE"),
    (_ok([_task(0, "search"), _task(1, "summary", "[5]")], "[1]"), "FUTURE_REFERENCE"),
    (_ok([_task(0, "search"), _task(1, "summary", "[0,0]")], "[1]"), "DUPLICATE_SOURCE"),
    (_ok([_task(0, "summary", '["user","user"]')]), "DUPLICATE_SOURCE"),
    (_ok([_task(0, "search", '["user"]')]), "SOURCE_COUNT"),
    (_ok([_task(0, "search"), _task(1, "search", "[0]")], "[1]"), "SOURCE_COUNT"),
    (_ok([_task(0, "sql", '["user"]')]), "SOURCE_COUNT"),
    (_ok([_task(0, "summary")]), "SOURCE_COUNT"),
    (_ok([_task(0, "extract")]), "SOURCE_COUNT"),
    (_ok([_task(0, "compare")]), "SOURCE_COUNT"),
    (_ok([SUMMARY], "[1]"), "INVALID_OUTPUT"),
    (_ok([SUMMARY], "[-1]"), "INVALID_OUTPUT"),
    (_ok([SUMMARY], "[0,0]"), "DUPLICATE_OUTPUT"),
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text")
    parser.add_argument("--mode", default="canonical", choices=["runtime", "canonical"])
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.text, mode=arguments.mode).to_dict(), ensure_ascii=False, indent=2))
