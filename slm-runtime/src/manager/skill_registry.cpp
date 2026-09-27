#include "manager/skill_registry.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace slm {

const char* skill_backend_kind_name(SkillBackendKind kind) noexcept {
    switch (kind) {
        case SkillBackendKind::Model: return "model";
        case SkillBackendKind::Retrieval: return "retrieval";
        case SkillBackendKind::Database: return "database";
    }
    return "model";
}

SkillRegistry::SkillRegistry(std::vector<SkillSpec> skills)
    : skills_(std::move(skills)) {
    if (skills_.empty()) {
        throw std::invalid_argument("skill registry must not be empty");
    }
    std::unordered_set<std::string> seen;
    for (const auto& skill : skills_) {
        if (skill.name.empty() || !seen.insert(skill.name).second) {
            throw std::invalid_argument("skill names must be unique and non-empty");
        }
        if (skill.min_sources > skill.max_sources) {
            throw std::invalid_argument("skill " + skill.name +
                                        " has min_sources > max_sources");
        }
        if (skill.backend == SkillBackendKind::Model &&
            skill.adapter_contract.empty()) {
            throw std::invalid_argument("model skill " + skill.name +
                                        " needs an adapter contract");
        }
    }
}

const SkillRegistry& SkillRegistry::manager_v1() {
    static const SkillRegistry registry({
        {"search",
         "External information retrieval. The instruction states what to look "
         "up. Takes no sources.",
         SkillBackendKind::Retrieval, 0, 0, ""},
        {"summary",
         "Condenses the content of its sources (user content or earlier task "
         "results) into a summary.",
         SkillBackendKind::Model, 1, kMaxTaskSources,
         "Summarize the source material below. Follow the task instruction. "
         "Use only the source material; do not add outside facts."},
        {"extract",
         "Pulls only the specific information named in the instruction out of "
         "its sources.",
         SkillBackendKind::Model, 1, kMaxTaskSources,
         "Extract only the information requested by the task instruction from "
         "the source material below. Return nothing that is not in the "
         "sources."},
        {"compare",
         "Compares targets, or task results, by the criteria in the "
         "instruction. Usually several sources; one source when a single "
         "document contains all targets.",
         SkillBackendKind::Model, 1, kMaxTaskSources,
         "Compare the targets named in the task instruction using only the "
         "source material below. State the comparison criteria and the "
         "differences."},
        {"sql",
         "Natural-language database request answered by the SQL skill, which "
         "owns schema, SQL generation and execution. Takes no sources.",
         SkillBackendKind::Database, 0, 0, ""},
    });
    return registry;
}

const SkillSpec* SkillRegistry::find(const std::string& name) const noexcept {
    for (const auto& skill : skills_) {
        if (skill.name == name) return &skill;
    }
    return nullptr;
}

const SkillSpec& SkillRegistry::at(const std::string& name) const {
    if (const auto* skill = find(name)) return *skill;
    throw std::invalid_argument("unsupported skill: '" + name + "'");
}

bool SkillRegistry::contains(const std::string& name) const noexcept {
    return find(name) != nullptr;
}

std::vector<std::string> SkillRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(skills_.size());
    for (const auto& skill : skills_) result.push_back(skill.name);
    return result;
}

}  // namespace slm
