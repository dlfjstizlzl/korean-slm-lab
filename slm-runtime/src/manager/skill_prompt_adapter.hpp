#pragma once

#include "manager/skill_registry.hpp"

#include <string>
#include <vector>

namespace slm {

// A task source after resolution: "user" or "task:N" plus its text.
struct ResolvedSource {
    std::string label;
    std::string text;
};

// Builds the prompt for a Model-backed skill (summary / extract / compare)
// from the registry's adapter contract, the Manager-written instruction and
// the resolved sources. Retrieval and Database skills never get a prompt
// here; they are dispatched to their backends by TaskExecutor.
class SkillPromptAdapter {
public:
    explicit SkillPromptAdapter(
        const SkillRegistry& registry = SkillRegistry::manager_v1());

    std::string build_prompt(const std::string& skill,
                             const std::string& instruction,
                             const std::vector<ResolvedSource>& sources) const;

private:
    const SkillRegistry* registry_;
};

}  // namespace slm
