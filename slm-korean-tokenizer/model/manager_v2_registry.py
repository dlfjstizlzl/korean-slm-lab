#!/usr/bin/env python3
"""Manager v1 / TaskPlan v2 contract, read from the Runtime itself.

The C++ `SkillRegistry::manager_v1()` is the single source of truth. Instead of
re-parsing C++ with regular expressions, this module runs the Runtime's
`slm_taskplan_tool --contract`, which prints the registry, limits, error codes,
the generated system prompt, the generated grammar and the few-shots, all from
the same compiled code that the Manager uses.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT.parent / "slm-runtime"
BUILD_DIR = RUNTIME_ROOT / "build"
TOOL = BUILD_DIR / "slm_taskplan_tool"
HASHED_SOURCES = (
    "src/manager/skill_registry.hpp",
    "src/manager/skill_registry.cpp",
    "src/manager/task_plan.hpp",
    "src/manager/task_plan.cpp",
    "src/manager/manager_client.hpp",
    "src/manager/manager_client.cpp",
    "src/manager/skill_prompt_adapter.cpp",
    "src/manager/task_executor.cpp",
    "src/skills/search_backend.cpp",
    "src/skills/sql_backend.cpp",
    "tools/taskplan_tool.cpp",
)


@dataclass(frozen=True)
class SkillContract:
    name: str
    description: str
    backend: str
    min_sources: int
    max_sources: int
    adapter_contract: str


@dataclass(frozen=True)
class ManagerContract:
    schema_version: int
    max_tasks: int
    max_task_sources: int
    skills: tuple[SkillContract, ...]
    statuses: tuple[str, ...]
    error_codes: tuple[str, ...]
    system_prompt: str
    grammar: str
    few_shots: tuple[dict, ...]
    source_sha256: dict = field(default_factory=dict)

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.skills)

    def skill(self, name: str) -> SkillContract | None:
        return next((skill for skill in self.skills if skill.name == name), None)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "max_tasks": self.max_tasks,
            "max_task_sources": self.max_task_sources,
            "skills": [skill.__dict__ for skill in self.skills],
            "statuses": list(self.statuses),
            "error_codes": list(self.error_codes),
            "few_shots": list(self.few_shots),
            "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode()).hexdigest(),
            "grammar_sha256": hashlib.sha256(self.grammar.encode()).hexdigest(),
            "source_sha256": self.source_sha256,
        }


def ensure_tool() -> Path:
    """Build the Runtime tool if needed (requires the existing CMake build dir)."""
    cmake = shutil.which("cmake")
    if cmake and (BUILD_DIR / "CMakeCache.txt").is_file():
        subprocess.run([cmake, "--build", str(BUILD_DIR), "--target", "slm_taskplan_tool", "-j8"],
                       check=True, capture_output=True)
    if not TOOL.is_file():
        raise FileNotFoundError(f"{TOOL} is missing; build slm-runtime first")
    return TOOL


@lru_cache(maxsize=1)
def load_contract() -> ManagerContract:
    tool = ensure_tool()
    raw = json.loads(subprocess.run([str(tool), "--contract"], check=True, capture_output=True,
                                    text=True).stdout)
    hashes = {name: hashlib.sha256((RUNTIME_ROOT / name).read_bytes()).hexdigest()
              for name in HASHED_SOURCES}
    return ManagerContract(
        schema_version=raw["schema_version"],
        max_tasks=raw["max_tasks"],
        max_task_sources=raw["max_task_sources"],
        skills=tuple(SkillContract(**skill) for skill in raw["skills"]),
        statuses=tuple(raw["statuses"]),
        error_codes=tuple(raw["error_codes"]),
        system_prompt=raw["system_prompt"],
        grammar=raw["grammar"],
        few_shots=tuple(raw["few_shots"]),
        source_sha256=hashes,
    )


if __name__ == "__main__":
    print(json.dumps(load_contract().to_dict(), ensure_ascii=False, indent=2))
