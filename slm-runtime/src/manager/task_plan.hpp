#pragma once

#include "manager/skill_registry.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace slm {

// TaskPlan v2 (Manager v1 canonical output).
//
//   {"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"...",
//     "sources":[]},{"id":1,"skill":"summary","instruction":"...",
//     "sources":[0]}],"outputs":[1]}
//   {"status":"clarify","message":"...","tasks":[],"outputs":[]}
//   {"status":"unsupported","message":"...","tasks":[],"outputs":[]}
inline constexpr int kTaskPlanSchemaVersion = 2;

enum class PlanStatus {
    Ok,
    Clarify,
    Unsupported,
};

const char* plan_status_name(PlanStatus status) noexcept;

// Every rejection carries a stable code. The Python dataset validator
// (slm-korean-tokenizer/model/manager_v2_validator.py) reports the same codes
// in the same check order; differential tests compare them.
enum class PlanErrorCode {
    NoJsonObject,
    JsonParseError,
    InvalidText,
    SchemaError,
    UnknownField,
    MissingField,
    TypeError,
    InvalidStatus,
    StatusSemantics,
    EmptyPlan,
    TooManyTasks,
    InvalidTaskId,
    InvalidSkill,
    EmptyInstruction,
    EmptyMessage,
    InvalidSource,
    SelfReference,
    FutureReference,
    DuplicateSource,
    SourceCount,
    EmptyOutputs,
    InvalidOutput,
    DuplicateOutput,
};

const char* plan_error_code_name(PlanErrorCode code) noexcept;

class TaskPlanError : public std::invalid_argument {
public:
    TaskPlanError(PlanErrorCode code, const std::string& message);
    PlanErrorCode code() const noexcept { return code_; }

private:
    PlanErrorCode code_;
};

// A source is either the user's original request/context or an earlier task.
struct SourceRef {
    static constexpr int kUser = -1;
    int task_id = kUser;

    static SourceRef user() { return SourceRef{kUser}; }
    static SourceRef task(int id) { return SourceRef{id}; }
    bool is_user() const noexcept { return task_id == kUser; }
    bool operator==(const SourceRef&) const = default;
};

struct Task {
    int id = 0;
    std::string skill;
    std::string instruction;
    std::vector<SourceRef> sources;
};

struct TaskPlan {
    PlanStatus status = PlanStatus::Ok;
    std::string message;  // clarify / unsupported only
    bool has_message = false;
    std::vector<Task> tasks;
    std::vector<int> outputs;

    static TaskPlan ok(std::vector<Task> tasks, std::vector<int> outputs);
    static TaskPlan clarify(std::string message);
    static TaskPlan unsupported(std::string message);

    // Parses the first '{' .. last '}' span of raw Manager output. Throws
    // TaskPlanError for syntax and structural problems.
    static TaskPlan from_manager_output(const std::string& output);

    // Semantic checks. Throws TaskPlanError on the first violation.
    void validate(const SkillRegistry& registry = SkillRegistry::manager_v1(),
                  std::size_t max_tasks = kMaxPlanTasks) const;

    // Compact canonical serialization; validates first.
    std::string to_json(
        const SkillRegistry& registry = SkillRegistry::manager_v1()) const;
};

// Text rule shared by instruction and message: UTF-8 already validated at
// parse time; rejects U+FFFD and control characters.
bool plan_text_is_invalid(const std::string& value);
// ASCII whitespace plus the Unicode space separators listed in task_plan.cpp.
bool plan_text_is_blank(const std::string& value);

}  // namespace slm
