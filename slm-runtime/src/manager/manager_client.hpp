#pragma once

#include "inference/inference_backend.hpp"
#include "manager/skill_registry.hpp"
#include "manager/task_plan.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace slm {

struct ManagerGeneration {
    GenerationResult generation;
    double request_latency_ms = 0.0;
};

struct ManagerMetrics {
    std::size_t attempts = 0;
    std::size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double latency_ms = 0.0;
    double decode_ms = 0.0;
    double decode_tokens_per_second = 0.0;
};

struct PlanningResult {
    TaskPlan plan;
    std::vector<std::string> raw_outputs;
    ManagerMetrics metrics;
};

struct ManagerFewShot {
    std::string family;  // provenance id, kept out of generated datasets
    std::string user;
    std::string plan_json;
};

class ManagerClient {
public:
    using Generator = std::function<ManagerGeneration(const std::string&)>;

    explicit ManagerClient(
        Generator generator,
        const SkillRegistry& registry = SkillRegistry::manager_v1());

    PlanningResult plan(const std::string& request) const;

    // All three are derived from the registry; nothing repeats skill names.
    static std::string build_system_prompt(const SkillRegistry& registry);
    static std::string build_json_grammar(const SkillRegistry& registry,
                                          std::size_t max_tasks = kMaxPlanTasks);
    static const std::vector<ManagerFewShot>& few_shots();

    static const std::string& system_prompt();  // manager_v1 registry
    static const std::string& json_grammar();   // manager_v1 registry

private:
    Generator generator_;
    const SkillRegistry* registry_;
};

}  // namespace slm
