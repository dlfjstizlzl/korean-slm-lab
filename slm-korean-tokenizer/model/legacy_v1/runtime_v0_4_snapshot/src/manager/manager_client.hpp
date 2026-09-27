#pragma once

#include "inference/inference_backend.hpp"
#include "manager/task_plan.hpp"

#include <functional>
#include <string>
#include <unordered_set>
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

class ManagerClient {
public:
    using Generator = std::function<ManagerGeneration(const std::string&)>;

    explicit ManagerClient(
        Generator generator,
        std::unordered_set<std::string> available_skills =
            {"search", "summary"});

    PlanningResult plan(const std::string& request) const;

    static const std::string& system_prompt();
    static const std::string& json_grammar();

private:
    Generator generator_;
    std::unordered_set<std::string> available_skills_;
};

}  // namespace slm
