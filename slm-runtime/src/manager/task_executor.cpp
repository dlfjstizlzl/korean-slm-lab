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

std::string source_label(const SourceRef& source) {
    return source.is_user() ? "user" : "task:" + std::to_string(source.task_id);
}

}  // namespace

std::string format_final_outputs(const std::vector<FinalOutput>& outputs) {
    if (outputs.size() == 1) return outputs.front().text;
    std::string text;
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        if (index > 0) text += "\n\n";
        text += "[output " + std::to_string(index + 1) + "/" +
                std::to_string(outputs.size()) + " · task " +
                std::to_string(outputs[index].task_id) + " · " +
                outputs[index].skill + "]\n" + outputs[index].text;
    }
    return text;
}

TaskExecutor::TaskExecutor(Runtime& runtime, SkillPromptAdapter adapter,
                           PrefetchMode prefetch_mode, SkillBackends backends,
                           const SkillRegistry& registry)
    : runtime_(runtime),
      adapter_(std::move(adapter)),
      prefetch_mode_(prefetch_mode),
      backends_(std::move(backends)),
      registry_(&registry) {}

ExecutionResult TaskExecutor::execute(const TaskPlan& plan,
                                      const RequestContext& context) {
    plan.validate(*registry_);
    const auto execution_start = std::chrono::steady_clock::now();
    ExecutionResult result;
    result.status = plan.status;
    if (plan.status != PlanStatus::Ok) {
        result.message = plan.message;
        result.final_output = plan.message;
        log_event("PLAN STATUS", std::string(plan_status_name(plan.status)) +
                                     " message=" + plan.message);
        result.total_ms = elapsed_ms(execution_start);
        return result;
    }
    result.tasks.reserve(plan.tasks.size());

    struct PendingPrefetch {
        std::string skill;
        std::shared_future<PrefetchResult> future;
        std::chrono::steady_clock::time_point work_started_at;
        std::chrono::steady_clock::time_point work_completed_at;
    };
    std::optional<PendingPrefetch> pending;

    auto next_model_prefetch = [&](std::size_t index) -> std::optional<std::string> {
        if (prefetch_mode_ != PrefetchMode::Plan || index + 1 >= plan.tasks.size()) {
            return std::nullopt;
        }
        const auto& next = plan.tasks[index + 1];
        if (registry_->at(next.skill).backend != SkillBackendKind::Model ||
            next.skill == plan.tasks[index].skill) {
            return std::nullopt;
        }
        return next.skill;
    };

    for (std::size_t index = 0; index < plan.tasks.size(); ++index) {
        const auto& task = plan.tasks[index];
        const SkillSpec& spec = registry_->at(task.skill);
        if (pending) {
            if (pending->skill != task.skill) {
                throw std::logic_error("prefetch target does not match next task");
            }
            const auto wait_started = std::chrono::steady_clock::now();
            const PrefetchResult prefetched = pending->future.get();
            const double wait_ms = elapsed_ms(wait_started);
            double overlap_ms = 0.0;
            if (prefetched.status == PrefetchStatus::Success) {
                const auto overlap_started =
                    std::max(prefetched.started_at, pending->work_started_at);
                const auto overlap_completed =
                    std::min(prefetched.completed_at, pending->work_completed_at);
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
            log_event("PREFETCH RESULT",
                      "skill=" + prefetched.skill + " status=" +
                          to_string(prefetched.status) + " load=" +
                          std::to_string(prefetched.load_ms) + "ms overlap=" +
                          std::to_string(overlap_ms) + "ms wait=" +
                          std::to_string(wait_ms) + "ms reason=" + prefetched.reason);
            pending.reset();
        }

        // Source resolution: "user" -> request context, N -> task N output.
        std::vector<ResolvedSource> sources;
        std::vector<std::string> labels;
        for (const auto& source : task.sources) {
            ResolvedSource resolved;
            resolved.label = source_label(source);
            if (source.is_user()) {
                resolved.text = context.user_content;
            } else {
                const auto source_id = static_cast<std::size_t>(source.task_id);
                if (source_id >= result.tasks.size()) {
                    throw std::logic_error("validated task source is unavailable");
                }
                resolved.text = result.tasks[source_id].output;
            }
            labels.push_back(resolved.label);
            sources.push_back(std::move(resolved));
        }
        std::string joined_labels;
        for (const auto& label : labels) {
            joined_labels += (joined_labels.empty() ? "" : ",") + label;
        }
        log_event("TASK", "id=" + std::to_string(task.id) + " skill=" + task.skill +
                              " backend=" + skill_backend_kind_name(spec.backend) +
                              " sources=[" + joined_labels + "]");

        TaskStepResult step;
        step.id = task.id;
        step.skill = task.skill;
        step.backend = skill_backend_kind_name(spec.backend);
        step.instruction = task.instruction;
        step.source_labels = labels;
        const auto prefetch_skill = next_model_prefetch(index);
        auto submit_prefetch = [&] {
            if (prefetch_skill) {
                pending = PendingPrefetch{*prefetch_skill,
                                          runtime_.prefetch(*prefetch_skill), {}, {}};
            }
        };

        if (spec.backend == SkillBackendKind::Model) {
            step.skill_input = adapter_.build_prompt(task.skill, task.instruction, sources);
            RuntimeRunResult run =
                runtime_.run_detailed(task.skill, step.skill_input, submit_prefetch);
            if (pending) {
                pending->work_started_at = run.inference_started_at;
                pending->work_completed_at = run.inference_completed_at;
            }
            step.output = run.generation.text;
            step.generation = std::move(run.generation);
            step.model_switch_ms = run.model_switch_ms;
            step.inference_ms = run.inference_ms;
            step.total_ms = run.total_ms;
        } else {
            step.skill_input = task.instruction;
            submit_prefetch();
            const auto started = std::chrono::steady_clock::now();
            if (spec.backend == SkillBackendKind::Retrieval) {
                SearchBackend* backend = backends_.search;
                if (backend == nullptr) {
                    throw SearchBackendUnavailable(
                        "search skill has no retrieval provider configured");
                }
                step.output = backend->search(task.instruction).to_text();
            } else {
                SqlBackend* backend = backends_.sql;
                if (backend == nullptr) {
                    throw SqlBackendUnavailable(
                        "sql skill has no database backend configured");
                }
                step.output = backend->run(task.instruction, backends_.sql_context).to_text();
            }
            if (pending) {
                pending->work_started_at = started;
                pending->work_completed_at = std::chrono::steady_clock::now();
            }
            step.inference_ms = elapsed_ms(started);
            step.total_ms = step.inference_ms;
        }
        result.model_switch_ms += step.model_switch_ms;
        result.skill_inference_ms += step.inference_ms;
        result.tasks.push_back(std::move(step));
    }

    for (const int output : plan.outputs) {
        const auto& step = result.tasks.at(static_cast<std::size_t>(output));
        result.final_outputs.push_back({step.id, step.skill, step.output});
    }
    result.final_output = format_final_outputs(result.final_outputs);
    result.total_ms = elapsed_ms(execution_start);
    result.visible_switch_ms = result.model_switch_ms + result.prefetch_wait_ms;
    result.peak_rss_bytes = process_peak_resident_bytes();
    return result;
}

}  // namespace slm
