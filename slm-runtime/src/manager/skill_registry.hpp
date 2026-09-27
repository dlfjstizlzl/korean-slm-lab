#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace slm {

inline constexpr std::size_t kMaxPlanTasks = 8;
// A task may read the user source plus every earlier task.
inline constexpr std::size_t kMaxTaskSources = kMaxPlanTasks + 1;

// How a skill is executed. Model skills run through Runtime/ModelManager
// (GGUF inference); Retrieval and Database skills call external backends.
enum class SkillBackendKind {
    Model,
    Retrieval,
    Database,
};

const char* skill_backend_kind_name(SkillBackendKind kind) noexcept;

struct SkillSpec {
    std::string name;
    // One line shown to the Manager in its system prompt.
    std::string description;
    SkillBackendKind backend = SkillBackendKind::Model;
    // Allowed number of entries in a task's `sources` array.
    std::size_t min_sources = 0;
    std::size_t max_sources = 0;
    // Model skills only: the fixed instruction SkillPromptAdapter places in
    // front of the Manager-written task instruction and the resolved sources.
    std::string adapter_contract;
};

// Manager v1 canonical skill catalog. This is the single source of truth for
// the grammar, the Manager system prompt, TaskPlan validation, the prompt
// adapter, the executor dispatch and main.cpp model registration.
class SkillRegistry {
public:
    explicit SkillRegistry(std::vector<SkillSpec> skills);

    static const SkillRegistry& manager_v1();

    const SkillSpec* find(const std::string& name) const noexcept;
    const SkillSpec& at(const std::string& name) const;
    bool contains(const std::string& name) const noexcept;
    const std::vector<SkillSpec>& skills() const noexcept { return skills_; }
    std::vector<std::string> names() const;

private:
    std::vector<SkillSpec> skills_;
};

}  // namespace slm
