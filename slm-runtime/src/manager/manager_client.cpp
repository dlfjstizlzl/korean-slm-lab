#include "manager/manager_client.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace slm {
namespace {

const char* kRoleAndRules = R"RULES(You are the Manager of a skill runtime. Convert the user request into one TaskPlan JSON object. You only plan. Never perform a task yourself: do not answer, search, summarize, extract, compare or write SQL in the plan.

TaskPlan:
{"status":"ok","tasks":[{"id":0,"skill":"<skill>","instruction":"<what the skill must do>","sources":[]}],"outputs":[0]}
{"status":"clarify","message":"<what the user must provide>","tasks":[],"outputs":[]}
{"status":"unsupported","message":"<why it cannot be done>","tasks":[],"outputs":[]}

Rules:
- status "ok": 1 to 8 tasks and at least one output. Task ids start at 0 and increase by 1.
- instruction: a short statement of what that skill must do, in the user's language. Never copy the user's long text into an instruction.
- sources: "user" refers to the user's own text in the request; an integer refers to an earlier task's result. Never refer to the same or a later task.
- Independent tasks appear in the order the request mentions them.
- outputs: the task ids whose results are returned to the user.
- status "clarify" when information the task needs is missing; status "unsupported" when no available skill can do it. Both have a message and empty tasks and outputs.
- Return compact JSON only.)RULES";

std::string skill_catalog(const SkillRegistry& registry) {
    std::ostringstream text;
    text << "Available skills:\n";
    for (const auto& skill : registry.skills()) {
        text << "- " << skill.name << ": " << skill.description;
        if (skill.max_sources == 0) {
            text << " sources must be [].";
        } else {
            text << " sources: at least " << skill.min_sources << ".";
        }
        text << '\n';
    }
    return text.str();
}

std::string retry_prompt(const std::string& request, const std::string& previous,
                         const std::string& error) {
    return "Original user request:\n" + request +
           "\n\nYour previous output was invalid.\nPrevious output:\n" + previous +
           "\nValidation error:\n" + error +
           "\nReturn a corrected valid TaskPlan JSON object only.";
}

std::string quoted(const std::string& value) {
    return "\"\\\"" + value + "\\\"\"";
}

std::string alternatives(std::size_t count) {
    std::string rule;
    for (std::size_t value = 0; value < count; ++value) {
        rule += (value ? " | \"" : "\"") + std::to_string(value) + "\"";
    }
    return rule;
}

}  // namespace

const std::vector<ManagerFewShot>& ManagerClient::few_shots() {
    static const std::vector<ManagerFewShot> examples = {
        {"fewshot.search", "Qwen3 최신 릴리스 소식 좀 알아봐 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"Qwen3 최신 릴리스 소식 검색","sources":[]}],"outputs":[0]})"},
        {"fewshot.summary_user",
         "아래 회의록 핵심만 정리해 줘. 회의록: 다음 주 배포는 수요일로 확정했다. QA는 월요일까지 회귀 테스트를 마친다. 디자인 리뷰는 금요일로 미룬다.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"summary","instruction":"회의록 핵심 내용 요약","sources":["user"]}],"outputs":[0]})"},
        {"fewshot.extract_user",
         "다음 공지에서 행사 날짜와 장소만 알려 줘: 올해 사내 해커톤은 11월 14일 판교 본사 3층 대회의실에서 열립니다.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"extract","instruction":"행사 날짜와 장소 추출","sources":["user"]}],"outputs":[0]})"},
        {"fewshot.sql", "지난 분기 지역별 신규 가입자 수를 조회해 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"sql","instruction":"지난 분기 지역별 신규 가입자 수 조회","sources":[]}],"outputs":[0]})"},
        {"fewshot.search_summary", "애플 M4 칩 성능 평가를 찾아서 짧게 정리해 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"애플 M4 칩 성능 평가 검색","sources":[]},{"id":1,"skill":"summary","instruction":"성능 평가 요약","sources":[0]}],"outputs":[1]})"},
        {"fewshot.search_extract", "이번 달 국제 유가 기사를 찾아서 가격 수치만 뽑아 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"이번 달 국제 유가 기사 검색","sources":[]},{"id":1,"skill":"extract","instruction":"유가 가격 수치 추출","sources":[0]}],"outputs":[1]})"},
        {"fewshot.search_search_compare", "Llama 3.1과 Mistral Large의 라이선스 조건을 비교해 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"Llama 3.1 라이선스 조건 검색","sources":[]},{"id":1,"skill":"search","instruction":"Mistral Large 라이선스 조건 검색","sources":[]},{"id":2,"skill":"compare","instruction":"라이선스 조건 비교","sources":[0,1]}],"outputs":[2]})"},
        {"fewshot.sql_summary", "어제 접수된 고객 문의를 조회해서 주요 불만을 요약해 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"sql","instruction":"어제 접수된 고객 문의 조회","sources":[]},{"id":1,"skill":"summary","instruction":"주요 불만 사항 요약","sources":[0]}],"outputs":[1]})"},
        {"fewshot.search_sql_compare", "갤럭시 버즈의 우리 매장 판매가와 온라인 최저가를 비교해 줘.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"sql","instruction":"갤럭시 버즈 매장 판매가 조회","sources":[]},{"id":1,"skill":"search","instruction":"갤럭시 버즈 온라인 최저가 검색","sources":[]},{"id":2,"skill":"compare","instruction":"판매가와 온라인 최저가 비교","sources":[0,1]}],"outputs":[2]})"},
        {"fewshot.multi_output",
         "이 보고서를 요약하고, 언급된 담당자 이름도 따로 뽑아 줘: 3분기 물류 비용은 8% 줄었다. 창고 이전은 김민수 팀장이, 운송 계약 재협상은 이지은 책임이 맡는다.",
         R"({"status":"ok","tasks":[{"id":0,"skill":"summary","instruction":"보고서 요약","sources":["user"]},{"id":1,"skill":"extract","instruction":"담당자 이름 추출","sources":["user"]}],"outputs":[0,1]})"},
        {"fewshot.clarify", "이거 요약해 줘.",
         R"({"status":"clarify","message":"요약할 내용을 제공해 주세요.","tasks":[],"outputs":[]})"},
        {"fewshot.unsupported", "이 문장을 일본어로 번역해 줘: 오늘 회의는 취소되었습니다.",
         R"({"status":"unsupported","message":"현재 지원하는 Skill로는 이 작업을 수행할 수 없습니다.","tasks":[],"outputs":[]})"},
    };
    return examples;
}

std::string ManagerClient::build_system_prompt(const SkillRegistry& registry) {
    std::ostringstream prompt;
    prompt << kRoleAndRules << "\n\n" << skill_catalog(registry) << "\nExamples:\n";
    for (const auto& example : few_shots()) {
        prompt << "User: " << example.user << "\nAssistant: " << example.plan_json << "\n";
    }
    std::string text = prompt.str();
    text.pop_back();
    return text;
}

std::string ManagerClient::build_json_grammar(const SkillRegistry& registry,
                                              std::size_t max_tasks) {
    std::ostringstream grammar;
    grammar << "root ::= ok-plan | message-plan\n";
    grammar << "message-plan ::= \"{\\\"status\\\":\" (" << quoted("clarify") << " | "
            << quoted("unsupported")
            << ") \",\\\"message\\\":\" text \",\\\"tasks\\\":[],\\\"outputs\\\":[]}\"\n";
    grammar << "ok-plan ::=";
    for (std::size_t count = 1; count <= max_tasks; ++count) {
        grammar << (count > 1 ? " |" : "") << " plan-" << count;
    }
    grammar << '\n';
    for (std::size_t count = 1; count <= max_tasks; ++count) {
        grammar << "plan-" << count << " ::= \"{\\\"status\\\":\\\"ok\\\",\\\"tasks\\\":[\"";
        for (std::size_t id = 0; id < count; ++id) {
            grammar << (id ? " \",\" " : " ") << "task-" << id;
        }
        grammar << " \"],\\\"outputs\\\":[\" out-" << count;
        if (count > 1) grammar << " (\",\" out-" << count << "){0," << count - 1 << "}";
        grammar << " \"]}\"\n";
        grammar << "out-" << count << " ::= " << alternatives(count) << '\n';
    }
    for (std::size_t id = 0; id < max_tasks; ++id) {
        grammar << "task-" << id << " ::= \"{\\\"id\\\":" << id << ",\\\"skill\\\":\" (";
        bool first = true;
        for (const auto& skill : registry.skills()) {
            // a skill that needs more sources than exist at this position
            // cannot appear here
            if (skill.min_sources > std::min(skill.max_sources, id + 1)) continue;
            grammar << (first ? "" : " | ") << skill.name << "-" << id;
            first = false;
        }
        grammar << ") \"}\"\n";
        grammar << "src-" << id << " ::= " << quoted("user");
        for (std::size_t source = 0; source < id; ++source) {
            grammar << " | \"" << source << "\"";
        }
        grammar << '\n';
        for (const auto& skill : registry.skills()) {
            const std::size_t reachable = std::min(skill.max_sources, id + 1);
            if (skill.min_sources > reachable) continue;
            grammar << skill.name << "-" << id << " ::= " << quoted(skill.name)
                    << " \",\\\"instruction\\\":\" text \",\\\"sources\\\":[";
            if (skill.max_sources == 0) {
                grammar << "]\"";
            } else if (skill.min_sources == 0) {
                grammar << "\" (src-" << id << " (\",\" src-" << id << "){0,"
                        << reachable - 1 << "})? \"]\"";
            } else {
                grammar << "\" src-" << id;
                if (reachable > 1) {
                    grammar << " (\",\" src-" << id << "){" << skill.min_sources - 1 << ","
                            << reachable - 1 << "}";
                }
                grammar << " \"]\"";
            }
            grammar << '\n';
        }
    }
    grammar << "text ::= \"\\\"\" text-first text-char* \"\\\"\"\n";
    grammar << "text-first ::= [^\"\\\\\\x00-\\x20\\x7F\\x80-\\xA0\\u1680\\u2000-\\u200A"
               "\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF\\uFFFD] | escape\n";
    grammar << "text-char ::= [^\"\\\\\\x00-\\x1F\\x7F\\x80-\\x9F\\uFFFD] | escape\n";
    grammar << "escape ::= \"\\\\\" [\"\\\\/]\n";
    return grammar.str();
}

const std::string& ManagerClient::system_prompt() {
    static const std::string prompt = build_system_prompt(SkillRegistry::manager_v1());
    return prompt;
}

const std::string& ManagerClient::json_grammar() {
    static const std::string grammar = build_json_grammar(SkillRegistry::manager_v1());
    return grammar;
}

ManagerClient::ManagerClient(Generator generator, const SkillRegistry& registry)
    : generator_(std::move(generator)), registry_(&registry) {
    if (!generator_) {
        throw std::invalid_argument("ManagerClient requires a generator");
    }
}

PlanningResult ManagerClient::plan(const std::string& request) const {
    if (request.empty()) {
        throw std::invalid_argument("manager request must not be empty");
    }

    log_event("MANAGER INPUT", "chars=" + std::to_string(request.size()));
    PlanningResult result;
    std::string prompt = request;
    std::string last_error;
    for (std::size_t attempt = 0; attempt < 2; ++attempt) {
        ManagerGeneration generated = generator_(prompt);
        result.raw_outputs.push_back(generated.generation.text);
        ++result.metrics.attempts;
        result.metrics.generated_tokens += generated.generation.generated_tokens;
        result.metrics.ttft_ms += generated.generation.ttft_ms;
        result.metrics.latency_ms += generated.request_latency_ms;
        result.metrics.decode_ms += generated.generation.decode_ms;
        log_event("MANAGER RAW", generated.generation.text);

        try {
            auto task_plan = TaskPlan::from_manager_output(generated.generation.text);
            task_plan.validate(*registry_);
            result.plan = std::move(task_plan);
            if (result.metrics.decode_ms > 0.0) {
                result.metrics.decode_tokens_per_second =
                    1000.0 * result.metrics.generated_tokens / result.metrics.decode_ms;
            }
            log_event("MANAGER PLAN",
                      std::string("status=") + plan_status_name(result.plan.status) +
                          " tasks=" + std::to_string(result.plan.tasks.size()) +
                          " outputs=" + std::to_string(result.plan.outputs.size()) +
                          " attempts=" + std::to_string(result.metrics.attempts));
            return result;
        } catch (const std::exception& error) {
            last_error = error.what();
            log_event("MANAGER ERROR", "reason=invalid_task_plan attempt=" +
                                           std::to_string(attempt + 1) +
                                           " error=" + last_error);
            if (attempt == 0) {
                prompt = retry_prompt(request, generated.generation.text, last_error);
            }
        }
    }
    throw std::runtime_error("manager produced an invalid task plan after one retry: " +
                             last_error);
}

}  // namespace slm
