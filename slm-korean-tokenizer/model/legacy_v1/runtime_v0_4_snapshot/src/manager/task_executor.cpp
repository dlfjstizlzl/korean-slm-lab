#include "manager/task_executor.hpp"

#include "core/logging.hpp"
#include "memory/macos/process_memory.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>
#include <utility>

namespace slm {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

TaskExecutor::TaskExecutor(Runtime& runtime, SkillPromptAdapter adapter,
                           PrefetchMode prefetch_mode)
    : runtime_(runtime),
      adapter_(std::move(adapter)),
      prefetch_mode_(prefetch_mode) {}

ExecutionResult TaskExecutor::execute(const TaskPlan& plan) {
    plan.validate();
    const auto execution_start = std::chrono::steady_clock::now();
    ExecutionResult result;
    result.tasks.reserve(plan.tasks.size());

    struct PendingPrefetch {
        std::string skill;
        std::shared_future<PrefetchResult> future;
        std::chrono::steady_clock::time_point inference_started_at;
        std::chrono::steady_clock::time_point inference_completed_at;
    };
    std::optional<PendingPrefetch> pending;

    for (std::size_t index = 0; index < plan.tasks.size(); ++index) {
        const auto& task = plan.tasks[index];
        if (pending) {
            if (pending->skill != task.skill) {
                throw std::logic_error("prefetch target does not match next task");
            }
            const auto wait_started = std::chrono::steady_clock::now();
            const PrefetchResult prefetched = pending->future.get();
            const double wait_ms = elapsed_ms(wait_started);
            double overlap_ms = 0.0;
            if (prefetched.status == PrefetchStatus::Success) {
                const auto overlap_started = std::max(
                    prefetched.started_at, pending->inference_started_at);
                const auto overlap_completed = std::min(
                    prefetched.completed_at, pending->inference_completed_at);
                if (overlap_completed > overlap_started) {
                    overlap_ms = std::chrono::duration<double, std::milli>(
                                     overlap_completed - overlap_started)
                                     .count();
                }
                overlap_ms = std::min(overlap_ms, prefetched.load_ms);
            }
            result.prefetch_time_ms += prefetched.load_ms;
            result.prefetch_overlap_ms += overlap_ms;
            result.prefetch_wait_ms += wait_ms;
            runtime_.record_prefetch_observation(overlap_ms, wait_ms);
            const double hidden_ratio = prefetched.load_ms > 0.0
                                            ? 100.0 * overlap_ms /
                                                  prefetched.load_ms
                                            : 0.0;
            log_event("PREFETCH RESULT",
                      "skill=" + prefetched.skill + " status=" +
                          to_string(prefetched.status) + " load=" +
                          std::to_string(prefetched.load_ms) +
                          "ms overlap=" + std::to_string(overlap_ms) +
                          "ms wait=" + std::to_string(wait_ms) +
                          "ms hidden_ratio=" +
                          std::to_string(hidden_ratio) + "% reason=" +
                          prefetched.reason);
            log_event("RSS", "stage=prefetch_completed skill=" + task.skill +
                                 " bytes=" +
                                 std::to_string(process_resident_bytes()));
            pending.reset();
        }

        std::string input;
        std::string source;
        if (task.input) {
            input = *task.input;
            source = "user";
        } else {
            const auto source_id = static_cast<std::size_t>(*task.input_from);
            if (source_id >= result.tasks.size()) {
                throw std::logic_error("validated task dependency is unavailable");
            }
            input = result.tasks[source_id].output;
            source = "task:" + std::to_string(*task.input_from);
        }

        log_event("TASK", "id=" + std::to_string(task.id) +
                              " skill=" + task.skill +
                              " input_source=" + source);
        const std::string prompt = adapter_.build_prompt(task.skill, input);
        const bool should_prefetch =
            prefetch_mode_ == PrefetchMode::Plan &&
            index + 1 < plan.tasks.size() &&
            plan.tasks[index + 1].skill != task.skill;
        RuntimeRunResult run = runtime_.run_detailed(
            task.skill, prompt, [&] {
                log_event("RSS", "stage=" + task.skill + "_hot bytes=" +
                                     std::to_string(process_resident_bytes()));
                if (should_prefetch) {
                    const std::string next_skill =
                        plan.tasks[index + 1].skill;
                    pending = PendingPrefetch{
                        next_skill, runtime_.prefetch(next_skill), {}, {}};
                    log_event("RSS", "stage=" + task.skill +
                                         "_hot_prefetch_submitted next=" +
                                         next_skill + " bytes=" +
                                         std::to_string(
                                             process_resident_bytes()));
                }
            });
        if (pending) {
            pending->inference_started_at = run.inference_started_at;
            pending->inference_completed_at = run.inference_completed_at;
        }
        log_event("RSS", "stage=" + task.skill + "_released bytes=" +
                             std::to_string(process_resident_bytes()));

        TaskStepResult step;
        step.id = task.id;
        step.skill = task.skill;
        step.resolved_input = std::move(input);
        step.output = run.generation.text;
        step.generation = std::move(run.generation);
        step.model_switch_ms = run.model_switch_ms;
        step.inference_ms = run.inference_ms;
        step.total_ms = run.total_ms;
        result.model_switch_ms += step.model_switch_ms;
        result.skill_inference_ms += step.inference_ms;
        result.tasks.push_back(std::move(step));
    }

    result.final_output = result.tasks.back().output;
    result.total_ms = elapsed_ms(execution_start);
    result.visible_switch_ms = result.model_switch_ms + result.prefetch_wait_ms;
    result.peak_rss_bytes = process_peak_resident_bytes();
    return result;
}

}  // namespace slm
