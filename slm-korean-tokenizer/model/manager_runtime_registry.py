#!/usr/bin/env python3
"""Read the Manager contract straight out of the slm-runtime C++ sources.

The Manager dataset must never carry its own copy of the skill list. The
runtime has no machine-readable registry, so this module parses the places
where the C++ code actually decides which skills a TaskPlan may name:

* `src/manager/manager_client.cpp`  - the llama.cpp grammar `skill` rule, the
  system prompt's "Available skills" block, and its few-shot examples
* `src/manager/task_plan.hpp`       - `kMaxPlanTasks` and the default
  `available_skills` argument of `TaskPlan::validate`
* `src/manager/manager_client.hpp`  - `ManagerClient`'s default skills
* `src/manager/skill_prompt_adapter.cpp` - the skills that have an adapter
* `src/main.cpp`                    - the skills manager mode keeps registered

If any of these disagree the runtime itself is inconsistent, so loading fails
loudly instead of guessing which one is right.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Pilot v1 was built against slm-runtime v0.4 (2 skills, input/input_from).
# The live runtime moved to TaskPlan v2; this frozen, hash-verified snapshot
# keeps the v1 artifacts reproducible. v2 uses model/manager_v2_registry.py.
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "model" / "legacy_v1" / "runtime_v0_4_snapshot"

SOURCE_FILES = (
    "src/manager/manager_client.cpp",
    "src/manager/manager_client.hpp",
    "src/manager/task_plan.hpp",
    "src/manager/task_plan.cpp",
    "src/manager/skill_prompt_adapter.cpp",
    "src/manager/task_executor.cpp",
    "src/main.cpp",
)


class RegistryMismatch(RuntimeError):
    """The runtime sources disagree about the Manager contract."""


@dataclass(frozen=True)
class RuntimeRegistry:
    runtime_root: str
    skills: tuple[str, ...]
    skill_descriptions: dict[str, str]
    adapter_instructions: dict[str, str]
    max_tasks: int
    grammar_max_tasks: int
    grammar: str
    system_prompt: str
    few_shots: tuple[tuple[str, str], ...]
    source_sha256: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "runtime_root": self.runtime_root,
            "skills": list(self.skills),
            "skill_descriptions": self.skill_descriptions,
            "adapter_instructions": self.adapter_instructions,
            "max_tasks": self.max_tasks,
            "grammar_max_tasks": self.grammar_max_tasks,
            "few_shots": [list(pair) for pair in self.few_shots],
            "source_sha256": self.source_sha256,
        }


def _raw_string(source: str, delimiter: str) -> str:
    match = re.search(r'R"' + delimiter + r"\((.*?)\)" + delimiter + '"', source, re.S)
    if not match:
        raise RegistryMismatch(f"raw string {delimiter} not found")
    return match.group(1)


def _brace_set(text: str) -> list[str]:
    return re.findall(r'"([A-Za-z0-9_]+)"', text)


def _grammar_skills(grammar: str) -> list[str]:
    rule = re.search(r"^skill\s*::=(.*)$", grammar, re.M)
    if not rule:
        raise RegistryMismatch("grammar has no skill rule")
    return re.findall(r'"\\"([A-Za-z0-9_]+)\\""', rule.group(1))


def _grammar_max_tasks(grammar: str) -> int:
    rule = re.search(r"^root\s*::=(.*)$", grammar, re.M)
    repeat = re.search(r'task \(","\s*ws task\)\{0,(\d+)\}', rule.group(1)) if rule else None
    if not repeat:
        raise RegistryMismatch("grammar root has no bounded task repetition")
    return int(repeat.group(1)) + 1


def _prompt_skills(prompt: str) -> dict[str, str]:
    block = re.search(r"Available skills:\n(.*?)\n\n", prompt, re.S)
    if not block:
        raise RegistryMismatch("system prompt has no Available skills block")
    return dict(re.findall(r"^- ([A-Za-z0-9_]+): (.+)$", block.group(1), re.M))


def _few_shots(prompt: str) -> tuple[tuple[str, str], ...]:
    return tuple(re.findall(r"^User: (.+)\nAssistant: (.+)$", prompt, re.M))


def _adapter_skills(source: str) -> dict[str, str]:
    pattern = re.compile(r'if \(skill == "([A-Za-z0-9_]+)"\) \{\s*return ((?:"[^"]*"\s*)+)', re.S)
    result = {}
    for name, literal in pattern.findall(source):
        pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', literal)
        text = "".join(pieces).encode("utf-8").decode("unicode_escape")
        result[name] = text
    return result


def load_registry(runtime_root: str | Path = DEFAULT_RUNTIME_ROOT) -> RuntimeRegistry:
    root = Path(runtime_root).resolve()
    sources = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime source missing: {path}")
        sources[relative] = path.read_text(encoding="utf-8")

    client = sources["src/manager/manager_client.cpp"]
    prompt = _raw_string(client, "PROMPT")
    grammar = _raw_string(client, "GRAMMAR")

    grammar_skills = _grammar_skills(grammar)
    prompt_skills = _prompt_skills(prompt)

    plan_header = sources["src/manager/task_plan.hpp"]
    max_match = re.search(r"kMaxPlanTasks\s*=\s*(\d+)", plan_header)
    validate_default = re.search(r"available_skills\s*=\s*\{([^}]*)\}", plan_header)
    client_default = re.search(r"available_skills\s*=\s*\{([^}]*)\}",
                               sources["src/manager/manager_client.hpp"])
    main_filter = re.search(r'model\.skill != "(\w+)" && model\.skill != "(\w+)"',
                            sources["src/main.cpp"])
    adapter = _adapter_skills(sources["src/manager/skill_prompt_adapter.cpp"])
    if not (max_match and validate_default and client_default and main_filter):
        raise RegistryMismatch("could not locate every skill declaration in the runtime")

    views = {
        "grammar": sorted(grammar_skills),
        "system_prompt": sorted(prompt_skills),
        "TaskPlan::validate default": sorted(_brace_set(validate_default.group(1))),
        "ManagerClient default": sorted(_brace_set(client_default.group(1))),
        "main.cpp manager filter": sorted(main_filter.groups()),
        "SkillPromptAdapter": sorted(adapter),
    }
    distinct = {tuple(value) for value in views.values()}
    if len(distinct) != 1:
        raise RegistryMismatch(f"runtime skill declarations disagree: {views}")

    max_tasks = int(max_match.group(1))
    grammar_max = _grammar_max_tasks(grammar)
    if grammar_max != max_tasks:
        raise RegistryMismatch(
            f"grammar allows {grammar_max} tasks but kMaxPlanTasks is {max_tasks}")

    return RuntimeRegistry(
        runtime_root=str(root),
        skills=tuple(grammar_skills),
        skill_descriptions={name: prompt_skills[name] for name in grammar_skills},
        adapter_instructions={name: adapter[name] for name in grammar_skills},
        max_tasks=max_tasks,
        grammar_max_tasks=grammar_max,
        grammar=grammar,
        system_prompt=prompt,
        few_shots=_few_shots(prompt),
        source_sha256={name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                       for name, text in sources.items()},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(load_registry().to_dict(), ensure_ascii=False, indent=2))
