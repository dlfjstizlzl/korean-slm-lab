#pragma once

#include "cache/cache_policy.hpp"
#include "core/model.hpp"
#include "inference/inference_backend.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace slm {

struct RuntimeMetrics {
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t evictions = 0;
    std::size_t loaded_model_bytes = 0;
    std::size_t peak_loaded_model_bytes = 0;
    std::size_t estimated_resident_memory = 0;
    std::size_t total_bytes_read = 0;
    std::vector<double> load_latencies_ms;
    std::vector<double> context_creation_latencies_ms;
    std::vector<double> unload_latencies_ms;
    std::vector<double> prefetch_latencies_ms;
    std::size_t prefetch_requests = 0;
    std::size_t prefetch_hits = 0;
    std::size_t prefetch_successes = 0;
    std::size_t prefetch_skips = 0;
    std::size_t prefetch_failures = 0;
    double prefetch_load_ms = 0.0;
    double prefetch_overlap_ms = 0.0;
    double prefetch_wait_ms = 0.0;
    std::size_t soft_budget_peak_bytes = 0;
};

enum class PrefetchStatus {
    Hit,
    Success,
    Skipped,
    Failed,
};

const char* to_string(PrefetchStatus status) noexcept;

struct PrefetchResult {
    std::string skill;
    PrefetchStatus status = PrefetchStatus::Skipped;
    std::string reason;
    double load_ms = 0.0;
    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point completed_at;
    std::size_t rss_before = 0;
    std::size_t rss_after = 0;
    bool used_soft_budget = false;
};

class ModelManager;

class ModelHandle {
public:
    ModelHandle() = default;
    ~ModelHandle();

    ModelHandle(const ModelHandle&) = delete;
    ModelHandle& operator=(const ModelHandle&) = delete;
    ModelHandle(ModelHandle&& other) noexcept;
    ModelHandle& operator=(ModelHandle&& other) noexcept;

    const Model& model() const;
    explicit operator bool() const noexcept;
    void reset();

private:
    friend class ModelManager;
    ModelHandle(ModelManager* manager, std::shared_ptr<Model> model);

    ModelManager* manager_ = nullptr;
    std::shared_ptr<Model> model_;
};

class ModelManager {
public:
    ModelManager(std::size_t memory_budget,
                 std::size_t prefetch_reserve,
                 std::shared_ptr<MemoryBackend> memory_backend,
                 std::unique_ptr<CachePolicy> cache_policy,
                 InferenceBackend& inference_backend);
    ModelManager(std::size_t memory_budget,
                 std::shared_ptr<MemoryBackend> memory_backend,
                 std::unique_ptr<CachePolicy> cache_policy,
                 InferenceBackend& inference_backend);

    void register_model(const std::string& skill, const std::string& path,
                        bool pinned = false);

    ModelHandle acquire(const std::string& skill);
    void release(const std::string& skill);
    void evict(const std::string& skill);
    PrefetchResult prefetch(const std::string& skill);
    void record_prefetch_observation(double overlap_ms, double wait_ms);

    RuntimeMetrics metrics() const;
    ModelState model_state(const std::string& skill) const;
    std::size_t memory_budget() const noexcept;
    std::size_t prefetch_reserve() const noexcept;

private:
    std::shared_ptr<Model> find_model_locked(const std::string& skill) const;
    std::size_t file_size_locked(const std::string& skill) const;
    void ensure_capacity_locked(std::size_t required, const std::string& incoming);
    bool try_make_capacity_locked(std::size_t limit, std::size_t required,
                                  const std::string& incoming);
    struct LoadOutcome {
        double duration_ms = 0.0;
        ModelLoadResult backend;
    };
    LoadOutcome perform_load(const std::shared_ptr<Model>& model);
    void finish_load_locked(const std::shared_ptr<Model>& model,
                            const LoadOutcome& outcome, bool is_prefetch,
                            bool used_soft_budget);
    bool evict_locked(const std::string& skill, const std::string& reason);

    std::size_t memory_budget_;
    std::size_t prefetch_reserve_;
    std::shared_ptr<MemoryBackend> memory_backend_;
    std::unique_ptr<CachePolicy> cache_policy_;
    InferenceBackend& inference_backend_;
    std::unordered_map<std::string, std::shared_ptr<Model>> models_;
    std::unordered_map<std::string, std::size_t> file_sizes_;
    RuntimeMetrics metrics_;
    mutable std::mutex mutex_;
    std::condition_variable load_condition_;
    std::unordered_map<std::string, std::size_t> loading_;
    std::size_t reserved_loading_bytes_ = 0;
};

}  // namespace slm
