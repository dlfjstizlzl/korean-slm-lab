#include "cache/lru_cache.hpp"
#include "core/logging.hpp"
#include "core/runtime.hpp"
#include "inference/inference_backend.hpp"
#include "inference/llama_cpp_inference_backend.hpp"
#include "memory/macos/mac_memory_backend.hpp"
#include "memory/macos/process_memory.hpp"
#include "manager/manager_client.hpp"
#include "manager/task_executor.hpp"
#include "skills/search_backend.hpp"
#include "skills/sql_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t mib = 1024ULL * 1024ULL;

struct RegisteredModel {
    std::string skill;
    std::filesystem::path path;
};

struct CliOptions {
    std::string backend = "dummy";
    std::vector<RegisteredModel> models;
    std::size_t budget_mb = 1200;
    std::size_t prefetch_reserve_mb = 0;
    slm::PrefetchMode prefetch_mode = slm::PrefetchMode::Off;
    bool budget_was_set = false;
    std::optional<std::filesystem::path> manager_path;
    std::size_t manager_budget_mb = 1024;
    std::uint32_t manager_max_tokens = 256;
    std::string search_backend = "none";
    std::string sql_backend = "none";
    std::optional<std::filesystem::path> sql_schema_path;
    std::optional<std::filesystem::path> context_file;
    slm::GenerationOptions generation;
    std::string prompt = "Hello. Briefly explain what virtual memory is.";
    bool prompt_was_set = false;
    bool interactive = false;
    bool show_help = false;
};

std::string require_value(int& index, int argc, char* argv[],
                          const std::string& option) {
    if (++index >= argc) throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

std::uint32_t parse_u32(const std::string& value, const std::string& option) {
    std::size_t used = 0;
    const unsigned long parsed = std::stoul(value, &used);
    if (used != value.size() || parsed > UINT32_MAX) {
        throw std::invalid_argument("invalid value for " + option + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

int parse_int(const std::string& value, const std::string& option) {
    std::size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) {
        throw std::invalid_argument("invalid value for " + option + ": " + value);
    }
    return parsed;
}

float parse_float(const std::string& value, const std::string& option) {
    std::size_t used = 0;
    const float parsed = std::stof(value, &used);
    if (used != value.size()) {
        throw std::invalid_argument("invalid value for " + option + ": " + value);
    }
    return parsed;
}

RegisteredModel parse_model(const std::string& value) {
    const auto separator = value.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= value.size()) {
        throw std::invalid_argument(
            "--skill/--model must use skill=/path/model.gguf");
    }
    return {value.substr(0, separator), value.substr(separator + 1)};
}

CliOptions parse_cli(int argc, char* argv[]) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--backend") {
            options.backend = require_value(index, argc, argv, argument);
        } else if (argument == "--model" || argument == "--skill") {
            options.models.push_back(
                parse_model(require_value(index, argc, argv, argument)));
        } else if (argument == "--manager") {
            options.manager_path =
                require_value(index, argc, argv, argument);
        } else if (argument == "--manager-budget-mb") {
            options.manager_budget_mb = std::stoull(
                require_value(index, argc, argv, argument));
        } else if (argument == "--manager-max-tokens") {
            options.manager_max_tokens = parse_u32(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--search-backend") {
            options.search_backend = require_value(index, argc, argv, argument);
            if (options.search_backend != "none" && options.search_backend != "mock") {
                throw std::invalid_argument("--search-backend must be none or mock");
            }
        } else if (argument == "--sql-backend") {
            options.sql_backend = require_value(index, argc, argv, argument);
            if (options.sql_backend != "none" && options.sql_backend != "mock") {
                throw std::invalid_argument("--sql-backend must be none or mock");
            }
        } else if (argument == "--sql-schema") {
            options.sql_schema_path = require_value(index, argc, argv, argument);
        } else if (argument == "--context-file") {
            options.context_file = require_value(index, argc, argv, argument);
        } else if (argument == "--budget-mb") {
            options.budget_mb = std::stoull(
                require_value(index, argc, argv, argument));
            options.budget_was_set = true;
        } else if (argument == "--prefetch-reserve-mb") {
            options.prefetch_reserve_mb = std::stoull(
                require_value(index, argc, argv, argument));
        } else if (argument == "--prefetch") {
            const auto value = require_value(index, argc, argv, argument);
            if (value == "off") {
                options.prefetch_mode = slm::PrefetchMode::Off;
            } else if (value == "plan") {
                options.prefetch_mode = slm::PrefetchMode::Plan;
            } else {
                throw std::invalid_argument("--prefetch must be off or plan");
            }
        } else if (argument == "--context") {
            options.generation.context_size = parse_u32(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--max-tokens") {
            options.generation.max_tokens = parse_u32(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--temperature") {
            options.generation.temperature = parse_float(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--top-p") {
            options.generation.top_p = parse_float(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--seed") {
            options.generation.seed = parse_u32(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--threads") {
            options.generation.cpu_threads = parse_int(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--gpu-layers") {
            const auto value = require_value(index, argc, argv, argument);
            options.generation.gpu_layers =
                value == "auto" ? -1 : parse_int(value, argument);
        } else if (argument == "--prompt") {
            options.prompt = require_value(index, argc, argv, argument);
            options.prompt_was_set = true;
        } else if (argument == "--interactive") {
            options.interactive = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.backend != "dummy" && options.backend != "llama") {
        throw std::invalid_argument("--backend must be dummy or llama");
    }
    if (options.budget_mb == 0 || options.manager_budget_mb == 0 ||
        options.manager_max_tokens == 0) {
        throw std::invalid_argument("budgets and token limits must be greater than zero");
    }
    if (options.backend == "llama" && !options.budget_was_set) {
        options.budget_mb = 2048;
    }
    return options;
}

void print_help(const char* executable) {
    std::cout
        << "SLM Runtime v0.5\n\n"
        << "Manager routing:\n  " << executable
        << " --backend llama --manager /path/manager.gguf"
           " --skill search=/path/search.gguf"
           " --skill summary=/path/summary.gguf --budget-mb 500"
           " --prefetch plan --prefetch-reserve-mb 300"
           " --prompt \"Find concepts and summarize them.\"\n\n"
        << "Dummy benchmark:\n  " << executable << " --backend dummy\n\n"
        << "Single GGUF, CPU-only:\n  " << executable
        << " --backend llama --model search=/path/model.gguf --gpu-layers 0\n\n"
        << "Two skills, Metal and LRU switching:\n  " << executable
        << " --backend llama --model search=/path/model.gguf"
           " --model summary=/path/model.gguf --budget-mb 800"
           " --gpu-layers auto\n\n"
        << "Options:\n"
        << "  --backend dummy|llama\n"
        << "  --model skill=path       repeatable\n"
        << "  --skill skill=path       repeatable, preferred spelling\n"
        << "  --manager path           enable Manager SLM routing\n"
        << "  --manager-budget-mb N    separate pinned budget, default 1024\n"
        << "  --manager-max-tokens N   default 256\n"
        << "  --search-backend none|mock  retrieval provider for the search skill;\n"
        << "                           none (default) fails search tasks, mock is\n"
        << "                           a deterministic offline test double\n"
        << "  --sql-backend none|mock  database backend for the sql skill\n"
        << "  --sql-schema PATH        schema/context given to the sql backend only\n"
        << "  --context-file PATH      text appended to the request as user content\n"
        << "  --interactive            read requests until exit\n"
        << "  --budget-mb N            default: dummy=1200, llama=2048\n"
        << "  --prefetch off|plan      default off (v0.3 behavior)\n"
        << "  --prefetch-reserve-mb N  temporary soft-budget reserve, default 0\n"
        << "  --context N              default 2048\n"
        << "  --max-tokens N           default 128\n"
        << "  --temperature F          default 0.7\n"
        << "  --top-p F                default 0.9\n"
        << "  --seed N                 default 42\n"
        << "  --threads N              default 4\n"
        << "  --gpu-layers N|auto      default 0 (CPU-only)\n"
        << "  --prompt TEXT            one request; manager mode routes it\n";
}

std::filesystem::path default_dummy_directory() {
    if (std::filesystem::exists("../models/search.bin")) return "../models";
    return "models";
}

std::vector<RegisteredModel> default_dummy_models() {
    const auto directory = default_dummy_directory();
    std::vector<RegisteredModel> result;
    for (const auto* skill : {"search", "sql", "summary", "code"}) {
        result.push_back({skill, directory / (std::string(skill) + ".bin")});
    }
    return result;
}

std::vector<RegisteredModel> default_llama_models() {
    std::filesystem::path directory = "models";
    if (!std::filesystem::exists(directory / "Qwen3-0.6B-Q4_0.gguf") &&
        std::filesystem::exists("../models/Qwen3-0.6B-Q4_0.gguf")) {
        directory = "../models";
    }
    auto summary_path = directory / "gemma-3-270m-it-Q8_0.gguf";
    if (!std::filesystem::is_regular_file(summary_path)) {
        summary_path = directory / "gemma-3-270m-Q8_0.gguf";
    }
    const std::vector<RegisteredModel> candidates{
        {"search", directory / "Qwen3-0.6B-Q4_0.gguf"},
        {"summary", summary_path},
        {"code", directory / "SmolLM3-Q4_K_M.gguf"},
    };
    std::vector<RegisteredModel> result;
    for (const auto& candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate.path)) {
            result.push_back(candidate);
        }
    }
    return result;
}

bool contains_skill(const std::vector<RegisteredModel>& models,
                    const std::string& skill) {
    return std::any_of(models.begin(), models.end(), [&](const auto& model) {
        return model.skill == skill;
    });
}

std::vector<std::pair<std::string, std::string>> make_workload(
    const CliOptions& options, const std::vector<RegisteredModel>& models) {
    if (options.backend == "dummy" && contains_skill(models, "search") &&
        contains_skill(models, "summary") && contains_skill(models, "sql") &&
        contains_skill(models, "code")) {
        return {{"search", "example request for search"},
                {"search", "second request for search"},
                {"summary", "example request for summary"},
                {"sql", "example request for sql"},
                {"search", "third request for search"},
                {"code", "example request for code"},
                {"summary", "second request for summary"},
                {"sql", "second request for sql"},
                {"search", "fourth request for search"}};
    }
    if (contains_skill(models, "search") && contains_skill(models, "summary")) {
        return {
            {"search", "Extract the main search keywords from: Apple Silicon unified memory architecture"},
            {"search", "Find the key concepts in: virtual memory and page faults"},
            {"summary", "Summarize this in one sentence: Virtual memory lets a process use an address space larger than physical RAM."},
            {"search", "Extract keywords from: memory mapped GGUF model loading"},
            {"summary", "Summarize this in one sentence: LRU evicts the least recently used inactive model."},
        };
    }
    if (models.size() == 1) {
        return {{models.front().skill, options.prompt}};
    }
    std::vector<std::pair<std::string, std::string>> workload;
    for (const auto& model : models) {
        workload.emplace_back(model.skill, options.prompt);
    }
    return workload;
}

double average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

void print_report(const slm::RuntimeBenchmark& report) {
    const auto& metrics = report.model_manager;
    const std::size_t requests = metrics.cache_hits + metrics.cache_misses;
    const double hit_rate = requests == 0
                                ? 0.0
                                : 100.0 * metrics.cache_hits / requests;
    std::vector<double> ttft;
    std::vector<double> speeds;
    for (const auto& generation : report.generations) {
        ttft.push_back(generation.ttft_ms);
        speeds.push_back(generation.decode_tokens_per_second);
    }

    std::cout << "\n=== Runtime Benchmark ===\n\n"
              << "Requests: " << requests << '\n'
              << "Cache hits: " << metrics.cache_hits << '\n'
              << "Cache misses: " << metrics.cache_misses << '\n'
              << "Total model loads: " << metrics.load_latencies_ms.size()
              << '\n'
              << std::fixed << std::setprecision(1)
              << "Hit rate: " << hit_rate << "%\n"
              << std::setprecision(2)
              << "Average model load latency: "
              << average(metrics.load_latencies_ms) << " ms\n"
              << "Average context creation latency: "
              << average(metrics.context_creation_latencies_ms) << " ms\n"
              << "Average model unload latency: "
              << average(metrics.unload_latencies_ms) << " ms\n"
              << "Average request latency: "
              << average(report.request_latencies_ms) << " ms\n"
              << "Average model switch latency: "
              << average(report.model_switch_latencies_ms) << " ms\n"
              << "Average TTFT: " << average(ttft) << " ms\n"
              << "Average decode speed: " << average(speeds) << " tok/s\n"
              << "Peak loaded-model cost: "
              << static_cast<double>(metrics.peak_loaded_model_bytes) / mib
              << " MB\n"
              << "Prefetch requests: " << metrics.prefetch_requests << '\n'
              << "Prefetch hits: " << metrics.prefetch_hits << '\n'
              << "Prefetch successes: " << metrics.prefetch_successes << '\n'
              << "Prefetch skips: " << metrics.prefetch_skips << '\n'
              << "Prefetch failures: " << metrics.prefetch_failures << '\n'
              << "Prefetch load time: " << metrics.prefetch_load_ms << " ms\n"
              << "Prefetch overlap: " << metrics.prefetch_overlap_ms << " ms\n"
              << "Prefetch wait: " << metrics.prefetch_wait_ms << " ms\n"
              << "Soft-budget peak cost: "
              << static_cast<double>(metrics.soft_budget_peak_bytes) / mib
              << " MB\n"
              << "Evictions: " << metrics.evictions << '\n'
              << "Estimated bytes loaded: " << metrics.total_bytes_read << '\n'
              << "Current process RSS: "
              << static_cast<double>(slm::process_resident_bytes()) / mib
              << " MB\n"
              << "Process peak RSS: "
              << static_cast<double>(slm::process_peak_resident_bytes()) / mib
              << " MB\n";
}

std::string describe_sources(const slm::Task& task) {
    std::string text;
    for (const auto& source : task.sources) {
        if (!text.empty()) text += ",";
        text += source.is_user() ? "user" : std::to_string(source.task_id);
    }
    return "[" + text + "]";
}

bool run_managed_request(slm::ManagerClient& manager,
                         slm::TaskExecutor& executor,
                         const std::string& request) {
    const auto request_start = std::chrono::steady_clock::now();
    try {
        const slm::PlanningResult planning = manager.plan(request);
        const auto& plan = planning.plan;
        std::cout << "\n[PLAN] status=" << slm::plan_status_name(plan.status) << "\n";
        for (const auto& task : plan.tasks) {
            std::cout << task.id << ' ' << task.skill << " sources="
                      << describe_sources(task) << " instruction=" << task.instruction
                      << '\n';
        }
        if (plan.status == slm::PlanStatus::Ok) {
            std::cout << "outputs=";
            for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
                std::cout << (index ? "," : "") << plan.outputs[index];
            }
            std::cout << '\n';
        }

        const slm::ExecutionResult execution =
            executor.execute(plan, slm::RequestContext{request});
        const double total_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    request_start)
                                    .count();
        std::cout << "\n[REQUEST METRICS]"
                  << " manager_latency=" << planning.metrics.latency_ms << "ms"
                  << " manager_ttft=" << planning.metrics.ttft_ms << "ms"
                  << " manager_tokens="
                  << planning.metrics.generated_tokens
                  << " manager_decode="
                  << planning.metrics.decode_tokens_per_second << " tok/s"
                  << " model_switch_latency=" << execution.model_switch_ms
                  << "ms skill_inference_latency="
                  << execution.skill_inference_ms << "ms"
                  << " prefetch_time=" << execution.prefetch_time_ms << "ms"
                  << " prefetch_overlap=" << execution.prefetch_overlap_ms
                  << "ms prefetch_wait=" << execution.prefetch_wait_ms << "ms"
                  << " visible_switch_latency="
                  << execution.visible_switch_ms << "ms"
                  << " peak_rss=" << execution.peak_rss_bytes
                  << " total_latency=" << total_ms << "ms\n";
        if (execution.status == slm::PlanStatus::Ok) {
            std::cout << "\n[FINAL outputs=" << execution.final_outputs.size() << "]\n"
                      << execution.final_output << "\n";
        } else {
            std::cout << "\n[" << (execution.status == slm::PlanStatus::Clarify
                                       ? "CLARIFY"
                                       : "UNSUPPORTED")
                      << "]\n" << execution.message << "\n";
        }
        return true;
    } catch (const std::exception& error) {
        slm::log_event("MANAGER ERROR",
                       "reason=request_failed error=" +
                           std::string(error.what()));
        std::cout << "\n[FINAL ERROR]\n" << error.what() << "\n";
        return false;
    }
}

void print_manager_report(const slm::RuntimeBenchmark& report,
                          double manager_load_ms) {
    const auto& metrics = report.model_manager;
    std::cout << "\n=== Manager Benchmark ===\n\n"
              << std::fixed << std::setprecision(2)
              << "Manager load latency: " << manager_load_ms << " ms\n"
              << "Manager inference requests: "
              << report.generations.size() << '\n'
              << "Manager cache hits: " << metrics.cache_hits << '\n'
              << "Manager cache misses: " << metrics.cache_misses << '\n'
              << "Manager evictions: " << metrics.evictions << '\n'
              << "Manager current loaded cost: "
              << static_cast<double>(metrics.loaded_model_bytes) / mib
              << " MB\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.show_help) {
            print_help(argv[0]);
            return 0;
        }

        auto models = options.models;
        if (options.backend == "dummy" && models.empty()) {
            models = default_dummy_models();
        } else if (options.backend == "llama" && models.empty()) {
            models = default_llama_models();
        }
        if (options.manager_path) {
            // Only Model-backed registry skills get GGUF models. search and sql
            // are served by retrieval/database backends, never by a GGUF.
            const auto& registry = slm::SkillRegistry::manager_v1();
            models.erase(
                std::remove_if(models.begin(), models.end(), [&](const auto& model) {
                    const auto* spec = registry.find(model.skill);
                    const bool keep = spec != nullptr &&
                                      spec->backend == slm::SkillBackendKind::Model;
                    if (!keep) {
                        slm::log_event("SKILL",
                                       "ignoring model for '" + model.skill +
                                           "': not a model-backed Manager v1 skill");
                    }
                    return !keep;
                }),
                models.end());
        }
        if (models.empty() && !options.manager_path) {
            throw std::invalid_argument(
                "no models were registered; pass --skill skill=/path/model.gguf");
        }
        for (const auto& model : models) {
            if (!std::filesystem::is_regular_file(model.path)) {
                throw std::invalid_argument("model file not found: '" +
                                            model.path.string() + "'");
            }
        }

        std::cout << "[RSS] process_start="
                  << slm::process_resident_bytes() << " bytes\n";

        if (options.manager_path) {
            if (options.backend != "llama") {
                throw std::invalid_argument(
                    "--manager requires --backend llama");
            }
            if (!std::filesystem::is_regular_file(*options.manager_path)) {
                throw std::invalid_argument(
                    "manager model file not found: '" +
                    options.manager_path->string() + "'");
            }

            slm::GenerationOptions manager_options = options.generation;
            manager_options.context_size =
                std::max<std::uint32_t>(manager_options.context_size, 4096);
            manager_options.max_tokens = options.manager_max_tokens;
            manager_options.temperature = 0.0F;
            manager_options.top_p = 1.0F;
            manager_options.system_prompt = slm::ManagerClient::system_prompt();
            manager_options.grammar = slm::ManagerClient::json_grammar();

            slm::Runtime manager_runtime(
                options.manager_budget_mb * mib,
                std::make_shared<slm::MacMemoryBackend>(),
                std::make_unique<slm::LruCachePolicy>(),
                slm::make_llama_cpp_inference_backend(manager_options));
            manager_runtime.register_model(
                "manager", options.manager_path->string(), true);
            const auto manager_after_load = manager_runtime.benchmark();
            const double manager_load_ms =
                manager_after_load.model_manager.load_latencies_ms.empty()
                    ? 0.0
                    : manager_after_load.model_manager.load_latencies_ms.front();
            slm::log_event(
                "MANAGER LOAD",
                "model=" + options.manager_path->string() +
                    " load_time=" + std::to_string(manager_load_ms) +
                    "ms pinned=true budget_mb=" +
                    std::to_string(options.manager_budget_mb) +
                    " rss_after=" +
                    std::to_string(slm::process_resident_bytes()));

            slm::Runtime skill_runtime(
                options.budget_mb * mib,
                options.prefetch_reserve_mb * mib,
                std::make_shared<slm::MacMemoryBackend>(),
                std::make_unique<slm::LruCachePolicy>(),
                slm::make_llama_cpp_inference_backend(options.generation));
            for (const auto& model : models) {
                skill_runtime.register_model(model.skill, model.path.string());
                if (model.skill == "summary" &&
                    model.path.filename().string().find("-it-") ==
                        std::string::npos) {
                    slm::log_event(
                        "WARNING",
                        "summary model appears to be a base model without an embedded chat template; prefer gemma-3-270m-it-Q8_0.gguf when available");
                }
            }

            slm::ManagerClient manager([&manager_runtime](
                                           const std::string& prompt) {
                auto run = manager_runtime.run_detailed("manager", prompt);
                return slm::ManagerGeneration{std::move(run.generation),
                                              run.total_ms};
            });
            auto search_backend = options.search_backend == "mock"
                                      ? slm::make_mock_search_backend()
                                      : slm::make_unconfigured_search_backend();
            auto sql_backend = options.sql_backend == "mock"
                                   ? slm::make_mock_sql_backend()
                                   : slm::make_unconfigured_sql_backend();
            slm::SkillBackends backends;
            backends.search = search_backend.get();
            backends.sql = sql_backend.get();
            if (options.sql_schema_path) {
                std::ifstream schema(*options.sql_schema_path);
                if (!schema) {
                    throw std::invalid_argument("cannot read --sql-schema file");
                }
                std::ostringstream text;
                text << schema.rdbuf();
                backends.sql_context.schema_description = text.str();
            }
            std::string attached_context;
            if (options.context_file) {
                std::ifstream context(*options.context_file);
                if (!context) {
                    throw std::invalid_argument("cannot read --context-file");
                }
                std::ostringstream text;
                text << context.rdbuf();
                attached_context = text.str();
            }
            slm::log_event("SKILL BACKENDS",
                           std::string("search=") + search_backend->name() +
                               " sql=" + sql_backend->name());
            slm::TaskExecutor executor(skill_runtime, slm::SkillPromptAdapter{},
                                       options.prefetch_mode, backends);
            auto with_context = [&](const std::string& request) {
                return attached_context.empty() ? request
                                                : request + "\n\n" + attached_context;
            };

            bool all_succeeded = true;
            const bool interactive =
                options.interactive || !options.prompt_was_set;
            if (interactive) {
                std::cout << "\nSLM Runtime v0.5\n"
                             "Type 'exit' to quit.\n";
                std::string request;
                while (true) {
                    std::cout << "\n> " << std::flush;
                    if (!std::getline(std::cin, request) || request == "exit") {
                        break;
                    }
                    if (request.empty()) continue;
                    all_succeeded =
                        run_managed_request(manager, executor, with_context(request)) &&
                        all_succeeded;
                }
            } else {
                all_succeeded =
                    run_managed_request(manager, executor, with_context(options.prompt));
            }

            for (const auto& model : models) {
                skill_runtime.evict(model.skill);
            }
            print_report(skill_runtime.benchmark());
            print_manager_report(manager_runtime.benchmark(), manager_load_ms);
            std::cout << "Manager remains pinned through final reporting.\n"
                      << "Current process RSS with manager resident: "
                      << std::fixed << std::setprecision(2)
                      << static_cast<double>(slm::process_resident_bytes()) / mib
                      << " MB\n";
            return all_succeeded ? 0 : 1;
        }

        if (options.interactive) {
            throw std::invalid_argument("--interactive requires --manager");
        }

        std::unique_ptr<slm::InferenceBackend> inference =
            options.backend == "llama"
                ? slm::make_llama_cpp_inference_backend(options.generation)
                : slm::make_dummy_inference_backend();

        slm::Runtime runtime(options.budget_mb * mib,
                             std::make_shared<slm::MacMemoryBackend>(),
                             std::make_unique<slm::LruCachePolicy>(),
                             std::move(inference));
        for (const auto& model : models) {
            runtime.register_model(model.skill, model.path.string());
        }

        if (options.backend == "dummy" && contains_skill(models, "summary")) {
            (void)runtime.prefetch("summary").get();
        }

        for (const auto& [skill, prompt] : make_workload(options, models)) {
            std::cout << "\n[PROMPT] skill=" << skill << "\n" << prompt << '\n';
            const auto result = runtime.run(skill, prompt);
            std::cout << "[OUTPUT] " << result.text << '\n';
        }

        // Explicitly unload remaining models so RSS after unload is observable.
        for (const auto& model : models) {
            runtime.evict(model.skill);
        }
        print_report(runtime.benchmark());
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
