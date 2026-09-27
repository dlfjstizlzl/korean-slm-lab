#pragma once

#include "core/runtime.hpp"
#include "manager/skill_prompt_adapter.hpp"
#include "manager/skill_registry.hpp"
#include "manager/task_plan.hpp"
#include "skills/search_backend.hpp"
#include "skills/sql_backend.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace slm {

// What the "user" source resolves to: the user's original request, plus any
// attached context appended by the caller. Manager plans reference it by the
// literal source "user" and never copy it into the plan.
struct RequestContext {
    std::string user_content;
};

// Non-model skill backends. Null pointers behave like unconfigured backends
// and make search/sql tasks fail with a clear error.
struct SkillBackends {
    SearchBackend* search = nullptr;
    SqlBackend* sql = nullptr;
    SqlContext sql_context;
};

struct TaskStepResult {
    int id = 0;
    std::string skill;
    std::string backend;
    std::string instruction;
    std::vector<std::string> source_labels;
    std::string skill_input;  // model prompt, search query or sql instruction
    std::string output;
    GenerationResult generation;
    double model_switch_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
};

struct FinalOutput {
    int task_id = 0;
    std::string skill;
    std::string text;
};

struct ExecutionResult {
    PlanStatus status = PlanStatus::Ok;
    std::string message;  // clarify / unsupported
    std::vector<TaskStepResult> tasks;
    std::vector<FinalOutput> final_outputs;  // in `outputs` order
    std::string final_output;                // deterministic rendering
    double model_switch_ms = 0.0;
    double skill_inference_ms = 0.0;
    double total_ms = 0.0;
    double prefetch_time_ms = 0.0;
    double prefetch_overlap_ms = 0.0;
    double prefetch_wait_ms = 0.0;
    double visible_switch_ms = 0.0;
    std::size_t peak_rss_bytes = 0;
};

enum class PrefetchMode {
    Off,
    Plan,
};

// Deterministic single-string rendering of several final outputs.
std::string format_final_outputs(const std::vector<FinalOutput>& outputs);

class TaskExecutor {
public:
    explicit TaskExecutor(Runtime& runtime,
                          SkillPromptAdapter adapter = SkillPromptAdapter{},
                          PrefetchMode prefetch_mode = PrefetchMode::Off,
                          SkillBackends backends = {},
                          const SkillRegistry& registry = SkillRegistry::manager_v1());

    // Tasks run sequentially in array order; `sources` always point backwards
    // so array order is a valid topological order of the dependency DAG.
    ExecutionResult execute(const TaskPlan& plan, const RequestContext& context);

private:
    Runtime& runtime_;
    SkillPromptAdapter adapter_;
    PrefetchMode prefetch_mode_;
    SkillBackends backends_;
    const SkillRegistry* registry_;
};

}  // namespace slm
