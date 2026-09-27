#include "core/runtime.hpp"

#include "core/logging.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace slm {
namespace {

double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::unique_ptr<InferenceBackend> require_backend(
    std::unique_ptr<InferenceBackend> backend) {
    if (!backend) {
        throw std::invalid_argument("Runtime requires an inference backend");
    }
    return backend;
}

}  // namespace

Runtime::Runtime(std::size_t memory_budget,
                 std::size_t prefetch_reserve,
                 std::shared_ptr<MemoryBackend> memory_backend,
                 std::unique_ptr<CachePolicy> cache_policy,
                 std::unique_ptr<InferenceBackend> inference_backend)
    : inference_backend_(require_backend(std::move(inference_backend))),
      model_manager_(memory_budget, prefetch_reserve, std::move(memory_backend),
                     std::move(cache_policy), *inference_backend_),
      prefetcher_(model_manager_) {}

Runtime::Runtime(std::size_t memory_budget,
                 std::shared_ptr<MemoryBackend> memory_backend,
                 std::unique_ptr<CachePolicy> cache_policy,
                 std::unique_ptr<InferenceBackend> inference_backend)
    : Runtime(memory_budget, 0, std::move(memory_backend),
              std::move(cache_policy), std::move(inference_backend)) {}

void Runtime::register_model(const std::string& skill, const std::string& path,
                             bool pinned) {
    model_manager_.register_model(skill, path, pinned);
}

std::shared_future<PrefetchResult> Runtime::prefetch(const std::string& skill) {
    return prefetcher_.prefetch(skill);
}

void Runtime::record_prefetch_observation(double overlap_ms, double wait_ms) {
    model_manager_.record_prefetch_observation(overlap_ms, wait_ms);
}

void Runtime::evict(const std::string& skill) {
    model_manager_.evict(skill);
}

GenerationResult Runtime::run(const std::string& skill,
                              const std::string& prompt) {
    return run_detailed(skill, prompt).generation;
}

RuntimeRunResult Runtime::run_detailed(const std::string& skill,
                                       const std::string& prompt) {
    return run_detailed(skill, prompt, {});
}

RuntimeRunResult Runtime::run_detailed(
    const std::string& skill, const std::string& prompt,
    const std::function<void()>& on_model_ready) {
    const auto request_start = std::chrono::steady_clock::now();
    const auto switch_start = std::chrono::steady_clock::now();
    auto model = model_manager_.acquire(skill);
    const double switch_ms = milliseconds_since(switch_start);

    if (on_model_ready) on_model_ready();

    const auto inference_start = std::chrono::steady_clock::now();
    GenerationResult result = inference_backend_->run(model.model().skill(), prompt);
    const auto inference_completed = std::chrono::steady_clock::now();
    const double inference_ms = milliseconds_since(inference_start);
    const double request_ms = milliseconds_since(request_start);
    {
        std::lock_guard lock(metrics_mutex_);
        model_switch_latencies_ms_.push_back(switch_ms);
        request_latencies_ms_.push_back(request_ms);
        generations_.push_back(result);
    }
    log_event("RUN", skill + " duration=" + std::to_string(request_ms) + "ms");
    return RuntimeRunResult{std::move(result), switch_ms, inference_ms,
                            request_ms, inference_start,
                            inference_completed};
}

RuntimeBenchmark Runtime::benchmark() const {
    RuntimeBenchmark snapshot;
    snapshot.model_manager = model_manager_.metrics();
    std::lock_guard lock(metrics_mutex_);
    snapshot.request_latencies_ms = request_latencies_ms_;
    snapshot.model_switch_latencies_ms = model_switch_latencies_ms_;
    snapshot.generations = generations_;
    return snapshot;
}

}  // namespace slm
