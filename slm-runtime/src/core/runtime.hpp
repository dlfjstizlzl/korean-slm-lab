#pragma once

#include "core/model_manager.hpp"
#include "inference/inference_backend.hpp"
#include "prefetch/prefetcher.hpp"

#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace slm {

struct RuntimeBenchmark {
    RuntimeMetrics model_manager;
    std::vector<double> request_latencies_ms;
    std::vector<double> model_switch_latencies_ms;
    std::vector<GenerationResult> generations;
};

struct RuntimeRunResult {
    GenerationResult generation;
    double model_switch_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
    std::chrono::steady_clock::time_point inference_started_at;
    std::chrono::steady_clock::time_point inference_completed_at;
};

class Runtime {
public:
    Runtime(std::size_t memory_budget,
            std::size_t prefetch_reserve,
            std::shared_ptr<MemoryBackend> memory_backend,
            std::unique_ptr<CachePolicy> cache_policy,
            std::unique_ptr<InferenceBackend> inference_backend);

    void register_model(const std::string& skill, const std::string& path,
                        bool pinned = false);
    Runtime(std::size_t memory_budget,
            std::shared_ptr<MemoryBackend> memory_backend,
            std::unique_ptr<CachePolicy> cache_policy,
            std::unique_ptr<InferenceBackend> inference_backend);

    std::shared_future<PrefetchResult> prefetch(const std::string& skill);
    void record_prefetch_observation(double overlap_ms, double wait_ms);
    void evict(const std::string& skill);
    GenerationResult run(const std::string& skill, const std::string& prompt);
    RuntimeRunResult run_detailed(const std::string& skill,
                                  const std::string& prompt);
    RuntimeRunResult run_detailed(const std::string& skill,
                                  const std::string& prompt,
                                  const std::function<void()>& on_model_ready);
    RuntimeBenchmark benchmark() const;

private:
    // Destruction is reversed: prefetcher, manager, then inference backend.
    std::unique_ptr<InferenceBackend> inference_backend_;
    ModelManager model_manager_;
    Prefetcher prefetcher_;
    std::vector<double> request_latencies_ms_;
    std::vector<double> model_switch_latencies_ms_;
    std::vector<GenerationResult> generations_;
    mutable std::mutex metrics_mutex_;
};

}  // namespace slm
