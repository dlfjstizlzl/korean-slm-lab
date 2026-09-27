#include "manager/skill_prompt_adapter.hpp"

#include <stdexcept>

namespace slm {

SkillPromptAdapter::SkillPromptAdapter(const SkillRegistry& registry)
    : registry_(&registry) {}

std::string SkillPromptAdapter::build_prompt(
    const std::string& skill, const std::string& instruction,
    const std::vector<ResolvedSource>& sources) const {
    const SkillSpec& spec = registry_->at(skill);
    if (spec.backend != SkillBackendKind::Model) {
        throw std::invalid_argument(
            "skill '" + skill + "' is a " + skill_backend_kind_name(spec.backend) +
            " skill and has no model prompt");
    }
    if (instruction.empty()) {
        throw std::invalid_argument("skill instruction must not be empty");
    }
    if (sources.size() < spec.min_sources) {
        throw std::invalid_argument("skill '" + skill + "' needs at least " +
                                    std::to_string(spec.min_sources) + " source(s)");
    }
    std::string prompt = spec.adapter_contract + "\n\nTask instruction:\n" +
                         instruction + "\n";
    for (std::size_t index = 0; index < sources.size(); ++index) {
        prompt += "\nSource " + std::to_string(index + 1) + " (" +
                  sources[index].label + "):\n" + sources[index].text + "\n";
    }
    return prompt;
}

}  // namespace slm
