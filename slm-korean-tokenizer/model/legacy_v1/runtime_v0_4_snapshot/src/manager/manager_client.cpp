#include "manager/manager_client.hpp"

#include "core/logging.hpp"

#include <stdexcept>
#include <utility>

namespace slm {
namespace {

const std::string kSystemPrompt = R"PROMPT(You are a task routing model.

Your only job is to convert a user request into a JSON execution plan.

Available skills:
- search: Extracts keywords, concepts, or information from input text.
- summary: Summarizes input text concisely.

Return JSON only. Do not explain your answer. Do not use markdown. Do not answer the user's task directly.

Schema:
The root object has one tasks array. Each task is exactly one of these shapes:
{"id":0,"skill":"search","input":"text"}
{"id":1,"skill":"summary","input_from":0}

Rules:
- Use one to eight tasks.
- Task ids start at 0 and increase by 1.
- Each task has exactly one of input or input_from.
- input_from refers only to an earlier task.
- Use only search and summary.

Examples:
User: Extract keywords from virtual memory.
Assistant: {"tasks":[{"id":0,"skill":"search","input":"Extract keywords from virtual memory."}]}

User: Summarize this: Virtual memory maps virtual addresses to physical memory.
Assistant: {"tasks":[{"id":0,"skill":"summary","input":"Virtual memory maps virtual addresses to physical memory."}]}

User: Find the key concepts in Apple Silicon unified memory and summarize them.
Assistant: {"tasks":[{"id":0,"skill":"search","input":"Find the key concepts in Apple Silicon unified memory."},{"id":1,"skill":"summary","input_from":0}]}

User: Give me a concise summary of page faults.
Assistant: {"tasks":[{"id":0,"skill":"summary","input":"page faults"}]})PROMPT";

const std::string kJsonGrammar = R"GRAMMAR(root         ::= "{" ws "\"tasks\"" ws ":" ws "[" ws task ("," ws task){0,7} "]" ws "}" ws
task         ::= "{" ws "\"id\"" ws ":" ws integer "," ws "\"skill\"" ws ":" ws skill "," ws input-source "}" ws
input-source ::= "\"input\"" ws ":" ws string | "\"input_from\"" ws ":" ws integer
skill        ::= "\"search\"" ws | "\"summary\"" ws
integer      ::= ([0-9] | [1-9] [0-9]{0,2}) ws
string       ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
ws           ::= | " " | "\n" [ \t]{0,20}
)GRAMMAR";

std::string retry_prompt(const std::string& request,
                         const std::string& previous,
                         const std::string& error) {
    return "Original user request:\n" + request +
           "\n\nYour previous output was invalid.\nPrevious output:\n" +
           previous + "\nValidation error:\n" + error +
           "\nReturn a corrected valid TaskPlan JSON object only.";
}

}  // namespace

ManagerClient::ManagerClient(Generator generator,
                             std::unordered_set<std::string> available_skills)
    : generator_(std::move(generator)),
      available_skills_(std::move(available_skills)) {
    if (!generator_) {
        throw std::invalid_argument("ManagerClient requires a generator");
    }
    if (available_skills_.empty()) {
        throw std::invalid_argument("ManagerClient requires available skills");
    }
}

PlanningResult ManagerClient::plan(const std::string& request) const {
    if (request.empty()) {
        throw std::invalid_argument("manager request must not be empty");
    }

    log_event("MANAGER INPUT", request);
    PlanningResult result;
    std::string prompt = request;
    std::string last_error;
    for (std::size_t attempt = 0; attempt < 2; ++attempt) {
        ManagerGeneration generated = generator_(prompt);
        result.raw_outputs.push_back(generated.generation.text);
        ++result.metrics.attempts;
        result.metrics.generated_tokens +=
            generated.generation.generated_tokens;
        result.metrics.ttft_ms += generated.generation.ttft_ms;
        result.metrics.latency_ms += generated.request_latency_ms;
        result.metrics.decode_ms += generated.generation.decode_ms;
        log_event("MANAGER RAW", generated.generation.text);

        try {
            auto task_plan =
                TaskPlan::from_manager_output(generated.generation.text);
            task_plan.validate(available_skills_);
            result.plan = std::move(task_plan);
            if (result.metrics.decode_ms > 0.0) {
                result.metrics.decode_tokens_per_second =
                    1000.0 * result.metrics.generated_tokens /
                    result.metrics.decode_ms;
            }
            log_event("MANAGER PLAN",
                      "tasks=" + std::to_string(result.plan.tasks.size()) +
                          " attempts=" +
                          std::to_string(result.metrics.attempts));
            return result;
        } catch (const std::exception& error) {
            last_error = error.what();
            log_event("MANAGER ERROR",
                      "reason=invalid_task_plan attempt=" +
                          std::to_string(attempt + 1) + " error=" + last_error);
            if (attempt == 0) {
                prompt = retry_prompt(request, generated.generation.text,
                                      last_error);
            }
        }
    }
    throw std::runtime_error(
        "manager produced an invalid task plan after one retry: " + last_error);
}

const std::string& ManagerClient::system_prompt() { return kSystemPrompt; }

const std::string& ManagerClient::json_grammar() { return kJsonGrammar; }

}  // namespace slm
