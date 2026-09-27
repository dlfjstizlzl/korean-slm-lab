// TaskPlan v2 tool: exposes the Runtime's own Manager contract to tooling.
//
//   slm_taskplan_tool --contract   JSON: registry, limits, prompt, grammar, few-shots
//   slm_taskplan_tool --validate   stdin: hex-encoded Manager outputs, one per line
//                                  stdout: "OK <to_json>" | "ERR <CODE> <message>"
//   slm_taskplan_tool --grammar    stdin: hex-encoded texts, one per line
//                                  stdout: "ACCEPT" | "REJECT <reason>" using
//                                  llama.cpp's grammar engine and json_grammar()
//   slm_taskplan_tool --grammar-session
//                                  incremental grammar state for constrained
//                                  decoding outside llama.cpp (evaluation only):
//                                    N <sid>              new session
//                                    T <sid> <hex>,<hex>  which candidate byte strings
//                                                         may follow ("-" = end of
//                                                         generation); prints 0/1 per item
//                                    A <sid> <hex>        commit bytes
//                                    D <sid>              drop session
#include "manager/manager_client.hpp"
#include "manager/skill_registry.hpp"
#include "manager/task_plan.hpp"

#include "llama-grammar.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <exception>
#include <iostream>
#include <string>

namespace {

std::string decode_hex(const std::string& hex) {
    std::string bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t index = 0; index + 1 < hex.size(); index += 2) {
        bytes.push_back(static_cast<char>(std::stoi(hex.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

std::string json_string(const std::string& value) {
    std::string out = "\"";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (character < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(character));
                }
        }
    }
    return out + "\"";
}

int dump_contract() {
    const auto& registry = slm::SkillRegistry::manager_v1();
    std::cout << "{\"schema_version\":" << slm::kTaskPlanSchemaVersion
              << ",\"max_tasks\":" << slm::kMaxPlanTasks
              << ",\"max_task_sources\":" << slm::kMaxTaskSources << ",\"skills\":[";
    bool first = true;
    for (const auto& skill : registry.skills()) {
        std::cout << (first ? "" : ",") << "{\"name\":" << json_string(skill.name)
                  << ",\"description\":" << json_string(skill.description)
                  << ",\"backend\":" << json_string(slm::skill_backend_kind_name(skill.backend))
                  << ",\"min_sources\":" << skill.min_sources
                  << ",\"max_sources\":" << skill.max_sources
                  << ",\"adapter_contract\":" << json_string(skill.adapter_contract) << "}";
        first = false;
    }
    std::cout << "],\"statuses\":[\"ok\",\"clarify\",\"unsupported\"],\"error_codes\":[";
    for (int code = 0; code <= static_cast<int>(slm::PlanErrorCode::DuplicateOutput); ++code) {
        std::cout << (code ? "," : "")
                  << json_string(slm::plan_error_code_name(static_cast<slm::PlanErrorCode>(code)));
    }
    std::cout << "],\"system_prompt\":" << json_string(slm::ManagerClient::system_prompt())
              << ",\"grammar\":" << json_string(slm::ManagerClient::json_grammar())
              << ",\"few_shots\":[";
    first = true;
    for (const auto& example : slm::ManagerClient::few_shots()) {
        std::cout << (first ? "" : ",") << "{\"family\":" << json_string(example.family)
                  << ",\"user\":" << json_string(example.user)
                  << ",\"plan\":" << json_string(example.plan_json) << "}";
        first = false;
    }
    std::cout << "]}\n";
    return 0;
}

int validate_lines() {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto plan = slm::TaskPlan::from_manager_output(decode_hex(line));
            plan.validate();
            std::cout << "OK " << plan.to_json() << '\n';
        } catch (const slm::TaskPlanError& error) {
            std::cout << "ERR " << slm::plan_error_code_name(error.code()) << ' '
                      << error.what() << '\n';
        } catch (const std::exception& error) {
            std::cout << "ERR INTERNAL " << error.what() << '\n';
        }
    }
    return 0;
}

int grammar_lines() {
    const std::string& grammar_text = slm::ManagerClient::json_grammar();
    std::string line;
    while (std::getline(std::cin, line)) {
        llama_grammar* grammar = llama_grammar_init_impl(
            nullptr, grammar_text.c_str(), "root", false, nullptr, 0, nullptr, 0);
        if (grammar == nullptr) {
            std::cerr << "grammar failed to parse\n";
            return 2;
        }
        std::string verdict = "ACCEPT";
        try {
            llama_grammar_accept_str(*grammar, decode_hex(line));
            const bool complete = std::any_of(grammar->stacks.begin(), grammar->stacks.end(),
                                              [](const auto& stack) { return stack.empty(); });
            if (!complete) verdict = "REJECT incomplete";
        } catch (const std::exception& error) {
            verdict = std::string("REJECT ") + error.what();
        }
        llama_grammar_free_impl(grammar);
        std::string single_line = verdict.substr(0, verdict.find('\n'));
        std::cout << single_line << '\n';
    }
    return 0;
}

bool grammar_complete(const llama_grammar& grammar) {
    return std::any_of(grammar.stacks.begin(), grammar.stacks.end(),
                       [](const auto& stack) { return stack.empty(); });
}

// Verbatim mirror of the static decode_utf8() in llama.cpp/src/llama-grammar.cpp
// (commit f172be756), needed because llama_grammar_apply_impl requires a vocab.
std::pair<std::vector<uint32_t>, llama_partial_utf8> mirror_decode_utf8(
        const std::string& src, llama_partial_utf8 partial_start) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    const char* pos = src.c_str();
    std::vector<uint32_t> code_points;
    code_points.reserve(src.size() + 1);
    uint32_t value = partial_start.value;
    int n_remain = partial_start.n_remain;
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            code_points.push_back(0);
            return {std::move(code_points), llama_partial_utf8{0, -1}};
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }
    if (partial_start.n_remain > 0 && n_remain == 0) code_points.push_back(value);
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        n_remain = lookup[first_byte >> 4] - 1;
        if (n_remain < 0) {
            code_points.clear();
            code_points.push_back(0);
            return {std::move(code_points), llama_partial_utf8{0, n_remain}};
        }
        uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) code_points.push_back(value);
    }
    code_points.push_back(0);
    return {std::move(code_points), llama_partial_utf8{value, n_remain}};
}

// Same decision as llama_grammar_apply_impl for non-EOG tokens: decode the
// piece against the current partial UTF-8 state and reject it only if every
// stack rejects it (llama_grammar_reject_candidates).
std::string candidate_verdicts(llama_grammar& state, const std::vector<std::string>& pieces) {
    std::string verdicts(pieces.size(), '0');
    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> decoded;
    decoded.reserve(pieces.size());
    llama_grammar_candidates candidates;
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const auto& piece = pieces[index];
        if (piece == "\x1f-EOG") {
            verdicts[index] = grammar_complete(state) ? '1' : '0';
            continue;
        }
        if (piece.empty() || piece[0] == 0) continue;
        decoded.push_back(mirror_decode_utf8(piece, state.partial_utf8));
        candidates.push_back({index, decoded.back().first.data(), decoded.back().second, 0});
        verdicts[index] = '1';
    }
    if (candidates.empty()) return verdicts;
    const auto& rules = llama_grammar_get_rules(&state);
    auto& stacks = llama_grammar_get_stacks(&state);
    if (stacks.empty()) {
        for (const auto& candidate : candidates) verdicts[candidate.index] = '0';
        return verdicts;
    }
    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);
    for (std::size_t i = 1; i < stacks.size(); ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    for (const auto& reject : rejects) verdicts[reject.index] = '0';
    return verdicts;
}

int grammar_session() {
    const std::string& grammar_text = slm::ManagerClient::json_grammar();
    std::unordered_map<std::string, llama_grammar*> sessions;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command, sid, payload;
        input >> command >> sid >> payload;
        if (command == "N") {
            if (sessions.contains(sid)) llama_grammar_free_impl(sessions[sid]);
            sessions[sid] = llama_grammar_init_impl(nullptr, grammar_text.c_str(), "root",
                                                    false, nullptr, 0, nullptr, 0);
            std::cout << (sessions[sid] ? "OK" : "ERR") << '\n';
        } else if (command == "T") {
            std::vector<std::string> pieces;
            std::stringstream items(payload);
            std::string item;
            while (std::getline(items, item, ',')) {
                pieces.push_back(item == "-" ? std::string("\x1f-EOG") : decode_hex(item));
            }
            std::cout << candidate_verdicts(*sessions.at(sid), pieces) << '\n';
        } else if (command == "A") {
            try {
                llama_grammar_accept_str(*sessions.at(sid), decode_hex(payload));
                std::cout << "OK" << '\n';
            } catch (const std::exception& error) {
                std::cout << "ERR " << error.what() << '\n';
            }
        } else if (command == "D") {
            if (sessions.contains(sid)) {
                llama_grammar_free_impl(sessions[sid]);
                sessions.erase(sid);
            }
            std::cout << "OK" << '\n';
        } else {
            std::cout << "ERR unknown command" << '\n';
        }
        std::cout.flush();
    }
    for (auto& [_, grammar] : sessions) llama_grammar_free_impl(grammar);
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--contract") return dump_contract();
    if (mode == "--validate") return validate_lines();
    if (mode == "--grammar") return grammar_lines();
    if (mode == "--grammar-session") return grammar_session();
    std::cerr << "usage: slm_taskplan_tool --contract | --validate | --grammar | --grammar-session\n";
    return 64;
}
