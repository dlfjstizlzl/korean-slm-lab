// TaskPlan v2 / Manager v1 tests: parser, validator, serializer, grammar,
// prompt, executor source resolution, outputs, search/sql contracts, and an
// end-to-end orchestration smoke over the real ManagerClient + TaskExecutor
// path with a scripted Manager and mock search/sql backends.
#include "cache/lru_cache.hpp"
#include "core/runtime.hpp"
#include "inference/inference_backend.hpp"
#include "manager/manager_client.hpp"
#include "manager/skill_prompt_adapter.hpp"
#include "manager/skill_registry.hpp"
#include "manager/task_executor.hpp"
#include "manager/task_plan.hpp"
#include "memory/macos/mac_memory_backend.hpp"
#include "skills/search_backend.hpp"
#include "skills/sql_backend.hpp"

#include "llama-grammar.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::string code_of(const std::string& text) {
    try {
        const auto plan = slm::TaskPlan::from_manager_output(text);
        plan.validate();
        return "OK";
    } catch (const slm::TaskPlanError& error) {
        return slm::plan_error_code_name(error.code());
    }
}

void expect(const std::string& text, const std::string& code, const std::string& label) {
    const auto actual = code_of(text);
    require(actual == code, label + ": expected " + code + ", got " + actual);
}

bool grammar_accepts(const std::string& text) {
    llama_grammar* grammar = llama_grammar_init_impl(
        nullptr, slm::ManagerClient::json_grammar().c_str(), "root", false, nullptr, 0,
        nullptr, 0);
    if (grammar == nullptr) return false;
    bool accepted = true;
    try {
        llama_grammar_accept_str(*grammar, text);
        accepted = std::any_of(grammar->stacks.begin(), grammar->stacks.end(),
                               [](const auto& stack) { return stack.empty(); });
    } catch (const std::exception&) {
        accepted = false;
    }
    llama_grammar_free_impl(grammar);
    return accepted;
}

std::string task(int id, const std::string& skill, const std::string& sources,
                 const std::string& instruction = "지시") {
    return "{\"id\":" + std::to_string(id) + ",\"skill\":\"" + skill +
           "\",\"instruction\":\"" + instruction + "\",\"sources\":" + sources + "}";
}

std::string ok_plan(const std::vector<std::string>& tasks, const std::string& outputs) {
    std::string joined;
    for (const auto& item : tasks) joined += (joined.empty() ? "" : ",") + item;
    return "{\"status\":\"ok\",\"tasks\":[" + joined + "],\"outputs\":" + outputs + "}";
}

void test_parse_and_serialize() {
    const std::vector<std::string> canonical = {
        ok_plan({task(0, "search", "[]", "Qwen3 최신 정보 검색"),
                 task(1, "summary", "[0]", "주요 변경점 요약")},
                "[1]"),
        ok_plan({task(0, "summary", "[\"user\"]", "핵심 내용 요약")}, "[0]"),
        ok_plan({task(0, "search", "[]"), task(1, "search", "[]"),
                 task(2, "compare", "[0,1]", "성능과 메모리 요구량 비교")},
                "[2]"),
        ok_plan({task(0, "summary", "[\"user\"]"), task(1, "extract", "[\"user\"]")},
                "[0,1]"),
        ok_plan({task(0, "search", "[]"), task(1, "compare", "[\"user\",0]")}, "[1]"),
        "{\"status\":\"clarify\",\"message\":\"요약할 내용을 제공해 주세요.\",\"tasks\":[],\"outputs\":[]}",
        "{\"status\":\"unsupported\",\"message\":\"현재 지원하는 Skill로는 이 작업을 수행할 수 없습니다.\",\"tasks\":[],\"outputs\":[]}",
        ok_plan({task(0, "sql", "[]", "인용 \\\"따옴표\\\" 역슬래시 \\\\ 끝")}, "[0]"),
    };
    for (const auto& text : canonical) {
        const auto plan = slm::TaskPlan::from_manager_output(text);
        require(plan.to_json() == text, "roundtrip must be byte-identical: " + text);
        require(grammar_accepts(text), "grammar must accept canonical plan: " + text);
    }
    const std::string spaced =
        "Plan:\n{ \"status\" : \"ok\", \"tasks\": [ {\"id\":0, \"skill\":\"sql\", "
        "\"instruction\":\"지난달 매출 조회\", \"sources\":[] } ], \"outputs\": [0] }\ntrailing";
    const auto plan = slm::TaskPlan::from_manager_output(spaced);
    require(plan.to_json() ==
                ok_plan({task(0, "sql", "[]", "지난달 매출 조회")}, "[0]"),
            "whitespace and prose are tolerated; serialization is compact");
    require(!grammar_accepts(spaced), "grammar only admits the compact canonical form");
}

void test_rejections() {
    const auto summary = task(0, "summary", "[\"user\"]");
    expect("no json", "NO_JSON_OBJECT", "no object");
    expect(ok_plan({summary}, "[0]") + "{}", "JSON_PARSE_ERROR", "trailing content");
    expect("{\"status\":\"ok\",\"status\":\"ok\",\"tasks\":[],\"outputs\":[]}", "JSON_PARSE_ERROR",
           "duplicate keys");
    expect(ok_plan({"{\"id\":0.0,\"skill\":\"summary\",\"instruction\":\"x\",\"sources\":[\"user\"]}"}, "[0]"),
           "JSON_PARSE_ERROR", "float id");
    expect(ok_plan({"{\"id\":\"0\",\"skill\":\"summary\",\"instruction\":\"x\",\"sources\":[\"user\"]}"}, "[0]"),
           "TYPE_ERROR", "string id");
    expect(ok_plan({"{\"id\":true,\"skill\":\"summary\",\"instruction\":\"x\",\"sources\":[\"user\"]}"}, "[0]"),
           "TYPE_ERROR", "bool id");
    expect("[1]", "NO_JSON_OBJECT", "array root");
    expect("{\"tasks\":[{\"id\":0,\"skill\":\"search\",\"input\":\"x\"}]}", "MISSING_FIELD",
           "legacy v1 plan has no status");
    expect("{\"status\":\"ok\",\"tasks\":[{\"id\":0,\"skill\":\"summary\",\"input_from\":0}],\"outputs\":[0]}",
           "UNKNOWN_FIELD", "legacy input_from field");
    expect(ok_plan({summary}, "[0]").insert(1, "\"note\":1,"), "UNKNOWN_FIELD", "unknown root field");
    expect(ok_plan({"{\"id\":0,\"skill\":\"summary\",\"instruction\":\"x\",\"sources\":[\"user\"],\"why\":1}"}, "[0]"),
           "UNKNOWN_FIELD", "unknown task field");
    expect("{\"status\":\"ok\",\"tasks\":[],\"outputs\":[]}", "EMPTY_PLAN", "ok without tasks");
    expect(ok_plan({summary}, "[]"), "EMPTY_OUTPUTS", "ok without outputs");
    expect("{\"status\":\"ok\",\"message\":\"x\",\"tasks\":[" + summary + "],\"outputs\":[0]}",
           "STATUS_SEMANTICS", "ok with message");
    expect("{\"status\":\"done\",\"tasks\":[],\"outputs\":[]}", "INVALID_STATUS", "unknown status");
    expect("{\"status\":\"clarify\",\"tasks\":[],\"outputs\":[]}", "STATUS_SEMANTICS",
           "clarify without message");
    expect("{\"status\":\"clarify\",\"message\":\"x\",\"tasks\":[" + summary + "],\"outputs\":[0]}",
           "STATUS_SEMANTICS", "clarify with tasks");
    expect("{\"status\":\"unsupported\",\"message\":\"  \",\"tasks\":[],\"outputs\":[]}",
           "EMPTY_MESSAGE", "blank message");
    std::vector<std::string> nine;
    for (int id = 0; id < 9; ++id) nine.push_back(task(id, "search", "[]"));
    expect(ok_plan(nine, "[8]"), "TOO_MANY_TASKS", "nine tasks");
    expect(ok_plan({task(1, "summary", "[\"user\"]")}, "[0]"), "INVALID_TASK_ID", "id not index");
    expect(ok_plan({task(0, "translate", "[\"user\"]")}, "[0]"), "INVALID_SKILL", "sixth skill");
    expect(ok_plan({task(0, "code", "[]")}, "[0]"), "INVALID_SKILL", "code is not a skill");
    expect(ok_plan({task(0, "summary", "[\"user\"]", "")}, "[0]"), "EMPTY_INSTRUCTION", "empty");
    expect(ok_plan({task(0, "summary", "[\"user\"]", "　 ")}, "[0]"), "EMPTY_INSTRUCTION",
           "unicode blank");
    expect(ok_plan({task(0, "summary", "[\"user\"]", "깨짐�")}, "[0]"), "INVALID_TEXT", "U+FFFD");
    expect(ok_plan({task(0, "summary", "[\"user\"]", "a\\nb")}, "[0]"), "INVALID_TEXT",
           "escaped control character");
    expect(ok_plan({task(0, "summary", "[\"user\"]", "\xC3\x28")}, "[0]"), "INVALID_TEXT",
           "invalid UTF-8 bytes");
    expect(ok_plan({task(0, "summary", "[\"users\"]")}, "[0]"), "INVALID_SOURCE", "bad source string");
    expect(ok_plan({task(0, "summary", "[-1]")}, "[0]"), "INVALID_SOURCE", "negative source");
    expect(ok_plan({task(0, "summary", "[true]")}, "[0]"), "TYPE_ERROR", "bool source");
    expect(ok_plan({task(0, "search", "[]"), task(1, "summary", "[1]")}, "[1]"), "SELF_REFERENCE",
           "self");
    expect(ok_plan({task(0, "summary", "[1]"), task(1, "search", "[]")}, "[0]"), "FUTURE_REFERENCE",
           "future");
    expect(ok_plan({task(0, "search", "[]"), task(1, "summary", "[0,0]")}, "[1]"),
           "DUPLICATE_SOURCE", "duplicate task source");
    expect(ok_plan({task(0, "summary", "[\"user\",\"user\"]")}, "[0]"), "DUPLICATE_SOURCE",
           "duplicate user source");
    expect(ok_plan({task(0, "search", "[\"user\"]")}, "[0]"), "SOURCE_COUNT", "search takes no sources");
    expect(ok_plan({task(0, "sql", "[\"user\"]")}, "[0]"), "SOURCE_COUNT", "sql takes no sources");
    expect(ok_plan({task(0, "summary", "[]")}, "[0]"), "SOURCE_COUNT", "summary needs a source");
    expect(ok_plan({task(0, "extract", "[]")}, "[0]"), "SOURCE_COUNT", "extract needs a source");
    expect(ok_plan({task(0, "compare", "[]")}, "[0]"), "SOURCE_COUNT", "compare needs a source");
    expect(ok_plan({task(0, "compare", "[\"user\"]")}, "[0]"), "OK", "compare one user document");
    expect(ok_plan({summary}, "[1]"), "INVALID_OUTPUT", "output out of range");
    expect(ok_plan({summary}, "[0,0]"), "DUPLICATE_OUTPUT", "duplicate output");
    expect(ok_plan({summary}, "[\"0\"]"), "TYPE_ERROR", "string output");
}

void test_grammar_and_prompt() {
    llama_grammar* grammar = llama_grammar_init_impl(
        nullptr, slm::ManagerClient::json_grammar().c_str(), "root", false, nullptr, 0,
        nullptr, 0);
    require(grammar != nullptr, "llama.cpp must parse the generated grammar");
    if (grammar) llama_grammar_free_impl(grammar);

    for (const auto& example : slm::ManagerClient::few_shots()) {
        require(code_of(example.plan_json) == "OK", "few-shot must validate: " + example.family);
        require(slm::TaskPlan::from_manager_output(example.plan_json).to_json() ==
                    example.plan_json,
                "few-shot must be canonical: " + example.family);
        require(grammar_accepts(example.plan_json), "grammar must accept few-shot " + example.family);
    }
    require(!grammar_accepts(ok_plan({task(0, "summary", "[]")}, "[0]")),
            "grammar rejects summary without sources");
    require(!grammar_accepts(ok_plan({task(0, "search", "[\"user\"]")}, "[0]")),
            "grammar rejects search with sources");
    require(!grammar_accepts(ok_plan({task(1, "search", "[]")}, "[0]")),
            "grammar rejects ids that are not the array index");
    require(!grammar_accepts(ok_plan({task(0, "search", "[]"), task(1, "summary", "[1]")}, "[1]")),
            "grammar rejects self references");
    require(!grammar_accepts(ok_plan({task(0, "search", "[]")}, "[1]")),
            "grammar rejects outputs beyond the task count");
    require(!grammar_accepts(ok_plan({task(0, "search", "[]", " x")}, "[0]")),
            "grammar rejects instructions that start with a space");
    require(!grammar_accepts("{\"status\":\"clarify\",\"message\":\"x\",\"tasks\":[" +
                             task(0, "search", "[]") + "],\"outputs\":[0]}"),
            "grammar rejects clarify plans with tasks");

    const auto& prompt = slm::ManagerClient::system_prompt();
    for (const auto& name : slm::SkillRegistry::manager_v1().names()) {
        require(prompt.find("- " + name + ":") != std::string::npos, "prompt lists skill " + name);
    }
    for (const char* legacy : {"input_from", "\"input\"", "keywords", "Use only search and summary"}) {
        require(prompt.find(legacy) == std::string::npos,
                std::string("prompt must not keep legacy semantics: ") + legacy);
    }
    require(slm::SkillRegistry::manager_v1().names() ==
                std::vector<std::string>({"search", "summary", "extract", "compare", "sql"}),
            "Manager v1 catalog is exactly five skills");
}

struct Harness {
    std::filesystem::path directory;
    std::unique_ptr<slm::Runtime> runtime;
    std::unique_ptr<slm::SearchBackend> search = slm::make_mock_search_backend();
    std::unique_ptr<slm::SqlBackend> sql = slm::make_mock_sql_backend();

    explicit Harness(const std::string& name) {
        directory = std::filesystem::temp_directory_path() / name;
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        runtime = std::make_unique<slm::Runtime>(
            1 << 20, std::make_shared<slm::MacMemoryBackend>(),
            std::make_unique<slm::LruCachePolicy>(), slm::make_dummy_inference_backend());
        for (const auto& skill : slm::SkillRegistry::manager_v1().skills()) {
            if (skill.backend != slm::SkillBackendKind::Model) continue;
            const auto path = directory / (skill.name + ".bin");
            std::ofstream(path, std::ios::binary) << std::string(4096, '\0');
            runtime->register_model(skill.name, path.string());
        }
    }
    ~Harness() { std::filesystem::remove_all(directory); }

    slm::TaskExecutor executor(slm::SqlContext context = {}) {
        slm::SkillBackends backends;
        backends.search = search.get();
        backends.sql = sql.get();
        backends.sql_context = std::move(context);
        return slm::TaskExecutor(*runtime, slm::SkillPromptAdapter{}, slm::PrefetchMode::Off,
                                 backends);
    }
};

void test_executor_contracts() {
    Harness harness("slm-taskplan-v2-executor");
    auto executor = harness.executor(slm::SqlContext{"TABLE sales(product TEXT, amount INT)"});
    const std::string user_text = "사용자 원문: RTX 4090 24GB, RTX 4080 16GB";

    const auto compare_plan = slm::TaskPlan::from_manager_output(ok_plan(
        {task(0, "search", "[]", "Gemma 3 최신 정보 검색"),
         task(1, "sql", "[]", "지난달 매출 상위 제품 조회"),
         task(2, "compare", "[\"user\",0,1]", "가격 비교"),
         task(3, "extract", "[\"user\"]", "GPU 이름과 VRAM 추출")},
        "[2,3]"));
    const auto result = executor.execute(compare_plan, {user_text});
    require(result.tasks.size() == 4, "all tasks execute");
    require(result.tasks[0].backend == "retrieval" &&
                result.tasks[0].output.find("MOCK") != std::string::npos &&
                result.tasks[0].output.find("Gemma 3 최신 정보 검색") != std::string::npos,
            "search dispatches the instruction as the retrieval query");
    require(result.tasks[1].backend == "database" &&
                result.tasks[1].output.find("provided(") != std::string::npos,
            "sql receives runtime-owned schema context");
    require(compare_plan.to_json().find("TABLE sales") == std::string::npos,
            "schema context never enters the TaskPlan");
    const auto& compare_prompt = result.tasks[2].skill_input;
    require(compare_prompt.find("Source 1 (user):\n" + user_text) != std::string::npos &&
                compare_prompt.find("Source 2 (task:0)") != std::string::npos &&
                compare_prompt.find("Source 3 (task:1)") != std::string::npos,
            "compare receives user, task 0 and task 1 sources in order");
    require(result.tasks[3].source_labels == std::vector<std::string>({"user"}),
            "extract resolves the user source");
    require(result.final_outputs.size() == 2 && result.final_outputs[0].task_id == 2 &&
                result.final_outputs[1].task_id == 3,
            "outputs select and order the final results");
    require(result.final_output.find("[output 1/2 · task 2 · compare]") == 0 &&
                result.final_output.find("[output 2/2 · task 3 · extract]") != std::string::npos,
            "multiple outputs render deterministically");

    const auto clarify = executor.execute(
        slm::TaskPlan::clarify("요약할 내용을 제공해 주세요."), {"이거 요약해 줘"});
    require(clarify.status == slm::PlanStatus::Clarify && clarify.tasks.empty() &&
                clarify.final_output == "요약할 내용을 제공해 주세요.",
            "clarify returns the message and runs nothing");
    const auto unsupported = executor.execute(slm::TaskPlan::unsupported("불가"), {"번역"});
    require(unsupported.status == slm::PlanStatus::Unsupported && unsupported.tasks.empty(),
            "unsupported runs nothing");

    slm::TaskExecutor unconfigured(*harness.runtime);
    bool search_failed = false;
    try {
        (void)unconfigured.execute(
            slm::TaskPlan::ok({slm::Task{0, "search", "x", {}}}, {0}), {"x"});
    } catch (const slm::SearchBackendUnavailable&) {
        search_failed = true;
    }
    require(search_failed, "search without a provider fails loudly instead of inventing results");
    bool sql_failed = false;
    try {
        (void)unconfigured.execute(slm::TaskPlan::ok({slm::Task{0, "sql", "x", {}}}, {0}), {"x"});
    } catch (const slm::SqlBackendUnavailable&) {
        sql_failed = true;
    }
    require(sql_failed, "sql without a backend fails loudly");

    const slm::SkillPromptAdapter adapter;
    for (const char* skill : {"search", "sql"}) {
        bool threw = false;
        try {
            (void)adapter.build_prompt(skill, "x", {{"user", "y"}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, std::string("no model prompt exists for ") + skill);
    }
    for (const char* skill : {"summary", "extract", "compare"}) {
        const auto prompt = adapter.build_prompt(skill, "지시", {{"user", "본문"}});
        require(prompt.find("Extract the key concepts or keywords") == std::string::npos,
                "legacy keyword-extraction prompt is gone");
        require(prompt.find("Task instruction:\n지시") != std::string::npos &&
                    prompt.find("Source 1 (user):\n본문") != std::string::npos,
                std::string("adapter keeps instruction and payload separate for ") + skill);
    }
}

struct Scenario {
    std::string name;
    std::string request;
    std::string plan;
    std::string expected_status;
    std::vector<std::string> expected_skills;
    std::size_t expected_outputs;
};

void test_orchestration_smoke() {
    const std::string article =
        "\n\n요즘 노트북 GPU 비교 글: RTX 4070 Laptop은 VRAM 8GB, RTX 4090 Laptop은 16GB를 탑재했다.";
    const std::vector<Scenario> scenarios = {
        {"search_summary", "Qwen3 최신 정보를 찾아 요약해줘",
         ok_plan({task(0, "search", "[]", "Qwen3 최신 정보 검색"),
                  task(1, "summary", "[0]", "최신 정보 요약")}, "[1]"),
         "ok", {"search", "summary"}, 1},
        {"summary_user", "핵심만 요약해줘" + article,
         ok_plan({task(0, "summary", "[\"user\"]", "핵심 내용 요약")}, "[0]"), "ok", {"summary"}, 1},
        {"extract_user", "이 글에서 GPU 이름과 VRAM만 뽑아줘" + article,
         ok_plan({task(0, "extract", "[\"user\"]", "GPU 이름과 VRAM 추출")}, "[0]"), "ok",
         {"extract"}, 1},
        {"search_search_compare", "Qwen과 Gemma 최신 정보를 찾아 비교해줘",
         ok_plan({task(0, "search", "[]", "Qwen 최신 정보 검색"),
                  task(1, "search", "[]", "Gemma 최신 정보 검색"),
                  task(2, "compare", "[0,1]", "최신 정보 비교")}, "[2]"),
         "ok", {"search", "search", "compare"}, 1},
        {"sql_summary", "지난달 매출 상위 제품을 조회해서 요약해줘",
         ok_plan({task(0, "sql", "[]", "지난달 매출 상위 제품 조회"),
                  task(1, "summary", "[0]", "조회 결과 요약")}, "[1]"),
         "ok", {"sql", "summary"}, 1},
        {"search_sql_compare", "우리 DB의 평균 판매가와 인터넷 최저가를 비교해줘",
         ok_plan({task(0, "sql", "[]", "평균 판매가 조회"),
                  task(1, "search", "[]", "인터넷 최저가 검색"),
                  task(2, "compare", "[0,1]", "판매가와 최저가 비교")}, "[2]"),
         "ok", {"sql", "search", "compare"}, 1},
        {"clarify", "그거 요약해줘",
         "{\"status\":\"clarify\",\"message\":\"요약할 내용을 제공해 주세요.\",\"tasks\":[],\"outputs\":[]}",
         "clarify", {}, 0},
        {"unsupported", "이 문서를 PDF로 만들어서 팀장님께 메일로 보내줘",
         "{\"status\":\"unsupported\",\"message\":\"현재 지원하는 Skill로는 이 작업을 수행할 수 없습니다.\",\"tasks\":[],\"outputs\":[]}",
         "unsupported", {}, 0},
    };
    std::map<std::string, std::string> script;
    for (const auto& scenario : scenarios) script[scenario.request] = scenario.plan;

    Harness harness("slm-taskplan-v2-smoke");
    auto executor = harness.executor();
    // Scripted Manager: stands in for the (not yet trained) Manager-SLM. Every
    // other component is the real Runtime path.
    int generator_calls = 0;
    slm::ManagerClient manager([&](const std::string& prompt) {
        ++generator_calls;
        slm::ManagerGeneration generation;
        const auto found = script.find(prompt);
        generation.generation.text = found == script.end() ? "not a plan" : found->second;
        return generation;
    });
    for (const auto& scenario : scenarios) {
        const auto planning = manager.plan(scenario.request);
        const auto result = executor.execute(planning.plan, {scenario.request});
        std::vector<std::string> skills;
        for (const auto& step : result.tasks) skills.push_back(step.skill);
        const bool passed = slm::plan_status_name(result.status) == scenario.expected_status &&
                            skills == scenario.expected_skills &&
                            result.final_outputs.size() == scenario.expected_outputs &&
                            !result.final_output.empty() && grammar_accepts(scenario.plan);
        require(passed, "smoke scenario " + scenario.name);
        std::cout << "[SMOKE] " << scenario.name << " status="
                  << slm::plan_status_name(result.status) << " tasks=" << skills.size()
                  << " outputs=" << result.final_outputs.size() << " "
                  << (passed ? "PASS" : "FAIL") << '\n';
    }
    require(generator_calls == static_cast<int>(scenarios.size()),
            "each scenario is planned in one attempt");
    bool failed_cleanly = false;
    try {
        (void)manager.plan("unknown request");
    } catch (const std::runtime_error&) {
        failed_cleanly = true;
    }
    require(failed_cleanly && generator_calls == static_cast<int>(scenarios.size()) + 2,
            "invalid Manager output is retried once, then fails");
}

}  // namespace

int main() {
    try {
        test_parse_and_serialize();
        test_rejections();
        test_grammar_and_prompt();
        test_executor_contracts();
        test_orchestration_smoke();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures > 0) {
        std::cerr << failures << " of " << checks << " TaskPlan v2 checks failed\n";
        return 1;
    }
    std::cout << "All " << checks << " TaskPlan v2 checks passed\n";
    return 0;
}
