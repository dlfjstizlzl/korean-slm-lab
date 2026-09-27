#include "manager/skill_prompt_adapter.hpp"

#include <stdexcept>

namespace slm {

std::string SkillPromptAdapter::build_prompt(const std::string& skill,
                                              const std::string& input) const {
    if (input.empty()) {
        throw std::invalid_argument("skill input must not be empty");
    }
    if (skill == "search") {
        return "Extract the key concepts or keywords from the following input.\n"
               "Return only a concise list.\n\nInput:\n" +
               input;
    }
    if (skill == "summary") {
        return "Summarize the following input concisely in one sentence.\n\n"
               "Input:\n" +
               input;
    }
    throw std::invalid_argument("unsupported skill prompt adapter: '" + skill +
                                "'");
}

}  // namespace slm
