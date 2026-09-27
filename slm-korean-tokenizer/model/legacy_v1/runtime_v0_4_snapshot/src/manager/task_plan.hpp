#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace slm {

inline constexpr std::size_t kMaxPlanTasks = 8;

struct Task {
    int id = 0;
    std::string skill;
    std::optional<std::string> input;
    std::optional<int> input_from;
};

struct TaskPlan {
    std::vector<Task> tasks;

    static TaskPlan from_manager_output(const std::string& output);

    void validate(
        const std::unordered_set<std::string>& available_skills =
            {"search", "summary"},
        std::size_t max_tasks = kMaxPlanTasks) const;
    std::string to_json() const;
};

}  // namespace slm
