#include "core/model_manager.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace slm {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::string megabytes(std::size_t bytes) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(1)
           << static_cast<double>(bytes) / (1024.0 * 1024.0) << "MB";
    return output.str();
}

}  // namespace

const char* to_string(PrefetchStatus status) noexcept {
    switch (status) {
        case PrefetchStatus::Hit: return "hit";
        case PrefetchStatus::Success: return "success";
        case PrefetchStatus::Skipped: return "skipped";
        case PrefetchStatus::Failed: return "failed";
    }
    return "failed";
}

ModelHandle::ModelHandle(ModelManager* manager, std::shared_ptr<Model> model)
    : manager_(manager), model_(std::move(model)) {}

ModelHandle::~ModelHandle() { reset(); }

ModelHandle::ModelHandle(ModelHandle&& other) noexcept
    : manager_(std::exchange(other.manager_, nullptr)),
      model_(std::move(other.model_)) {}

ModelHandle& ModelHandle::operator=(ModelHandle&& other) noexcept {
    if (this != &other) {
        reset();
        manager_ = std::exchange(other.manager_, nullptr);
        model_ = std::move(other.model_);
    }
    return *this;
}

const Model& ModelHandle::model() const {
    if (!model_) throw std::logic_error("empty model handle");
    return *model_;
}

ModelHandle::operator bool() const noexcept { return model_ != nullptr; }

void ModelHandle::reset() {
    if (manager_ != nullptr && model_) manager_->release(model_->skill());
    manager_ = nullptr;
    model_.reset();
}

ModelManager::ModelManager(std::size_t memory_budget,
                           std::size_t prefetch_reserve,
                           std::shared_ptr<MemoryBackend> memory_backend,
                           std::unique_ptr<CachePolicy> cache_policy,
                           InferenceBackend& inference_backend)
    : memory_budget_(memory_budget),
      prefetch_reserve_(prefetch_reserve),
      memory_backend_(std::move(memory_backend)),
      cache_policy_(std::move(cache_policy)),
      inference_backend_(inference_backend) {
    if (memory_budget_ == 0 || !memory_backend_ || !cache_policy_) {
        throw std::invalid_argument("ModelManager requires a budget and backends");
    }
}

ModelManager::ModelManager(std::size_t memory_budget,
                           std::shared_ptr<MemoryBackend> memory_backend,
                           std::unique_ptr<CachePolicy> cache_policy,
                           InferenceBackend& inference_backend)
    : ModelManager(memory_budget, 0, std::move(memory_backend),
                   std::move(cache_policy), inference_backend) {}

void ModelManager::register_model(const std::string& skill,
                                  const std::string& path, bool pinned) {
    const std::size_t bytes = std::filesystem::file_size(path);
    if (bytes == 0) {
        throw std::invalid_argument("model file is empty: '" + path + "'");
    }
    if (bytes > memory_budget_) {
        throw std::invalid_argument("model '" + skill +
                                    "' exceeds the hard memory budget");
    }

    std::shared_ptr<Model> model;
    {
        std::unique_lock lock(mutex_);
        if (models_.contains(skill)) {
            throw std::invalid_argument("duplicate model skill: '" + skill + "'");
        }
        ModelDescriptor descriptor{skill, std::filesystem::path(path), bytes};
        model = std::make_shared<Model>(std::move(descriptor), pinned,
                                        memory_backend_);
        models_.emplace(skill, model);
        file_sizes_[skill] = bytes;
        if (!pinned) return;

        ensure_capacity_locked(bytes, skill);
        loading_[skill] = bytes;
        reserved_loading_bytes_ += bytes;
    }

    try {
        const auto outcome = perform_load(model);
        std::lock_guard lock(mutex_);
        reserved_loading_bytes_ -= bytes;
        loading_.erase(skill);
        finish_load_locked(model, outcome, false, false);
        if (!inference_backend_.owns_model_memory()) model->prefetch_dummy();
        cache_policy_->touch(skill);
        load_condition_.notify_all();
    } catch (...) {
        std::lock_guard lock(mutex_);
        reserved_loading_bytes_ -= bytes;
        loading_.erase(skill);
        load_condition_.notify_all();
        throw;
    }
}

ModelHandle ModelManager::acquire(const std::string& skill) {
    std::unique_lock lock(mutex_);
    const auto model = find_model_locked(skill);
    load_condition_.wait(lock, [&] { return !loading_.contains(skill); });

    if (model->is_loaded()) {
        if (!inference_backend_.is_loaded(skill)) {
            throw std::logic_error("model state disagrees with inference backend: '" +
                                   skill + "'");
        }
        if (!try_make_capacity_locked(memory_budget_, 0, skill)) {
            throw std::runtime_error("hard memory budget cannot activate model '" +
                                     skill + "'");
        }
        ++metrics_.cache_hits;
        log_event("MODEL HIT", skill + " state=" + to_string(model->state()));
    } else {
        ++metrics_.cache_misses;
        const std::size_t bytes = file_size_locked(skill);
        ensure_capacity_locked(bytes, skill);
        loading_[skill] = bytes;
        reserved_loading_bytes_ += bytes;
        lock.unlock();
        try {
            const auto outcome = perform_load(model);
            lock.lock();
            reserved_loading_bytes_ -= bytes;
            loading_.erase(skill);
            finish_load_locked(model, outcome, false, false);
            load_condition_.notify_all();
        } catch (...) {
            lock.lock();
            reserved_loading_bytes_ -= bytes;
            loading_.erase(skill);
            load_condition_.notify_all();
            throw;
        }
    }

    model->mark_hot();
    cache_policy_->touch(skill);
    return ModelHandle(this, model);
}

void ModelManager::release(const std::string& skill) {
    std::lock_guard lock(mutex_);
    const auto model = find_model_locked(skill);
    model->release();
    cache_policy_->touch(skill);
}

void ModelManager::evict(const std::string& skill) {
    std::unique_lock lock(mutex_);
    (void)find_model_locked(skill);
    load_condition_.wait(lock, [&] { return !loading_.contains(skill); });
    if (!find_model_locked(skill)->is_loaded()) return;
    if (!evict_locked(skill, "manual")) {
        throw std::logic_error("model cannot be evicted while HOT or PINNED: '" +
                               skill + "'");
    }
}

PrefetchResult ModelManager::prefetch(const std::string& skill) {
    PrefetchResult result;
    result.skill = skill;
    result.started_at = std::chrono::steady_clock::now();

    std::shared_ptr<Model> model;
    std::size_t bytes = 0;
    bool used_soft_budget = false;
    try {
        std::unique_lock lock(mutex_);
        ++metrics_.prefetch_requests;
        model = find_model_locked(skill);
        load_condition_.wait(lock, [&] { return !loading_.contains(skill); });
        if (model->is_loaded()) {
            ++metrics_.prefetch_hits;
            result.status = PrefetchStatus::Hit;
            result.reason = "already_loaded";
            result.completed_at = std::chrono::steady_clock::now();
            cache_policy_->touch(skill);
            log_event("PREFETCH HIT", "skill=" + skill + " state=" +
                                          to_string(model->state()));
            return result;
        }

        std::error_code path_error;
        if (!std::filesystem::is_regular_file(model->path(), path_error)) {
            ++metrics_.prefetch_skips;
            result.status = PrefetchStatus::Skipped;
            result.reason = "invalid_path";
            result.completed_at = std::chrono::steady_clock::now();
            log_event("PREFETCH SKIP", "skill=" + skill +
                                           " reason=invalid_path");
            return result;
        }

        bytes = file_size_locked(skill);
        if (!try_make_capacity_locked(memory_budget_, bytes, skill)) {
            const std::size_t soft_limit = memory_budget_ + prefetch_reserve_;
            if (prefetch_reserve_ == 0 ||
                !try_make_capacity_locked(soft_limit, bytes, skill)) {
                ++metrics_.prefetch_skips;
                result.status = PrefetchStatus::Skipped;
                result.reason = "budget";
                result.completed_at = std::chrono::steady_clock::now();
                log_event("PREFETCH SKIP", "skill=" + skill +
                                               " reason=budget hard=" +
                                               megabytes(memory_budget_) +
                                               " reserve=" +
                                               megabytes(prefetch_reserve_));
                return result;
            }
            used_soft_budget = true;
        }
        loading_[skill] = bytes;
        reserved_loading_bytes_ += bytes;
        result.used_soft_budget = used_soft_budget;
        result.started_at = std::chrono::steady_clock::now();
        log_event("PREFETCH START", "skill=" + skill + " size=" +
                                        megabytes(bytes) + " budget=" +
                                        (used_soft_budget ? "soft" : "hard"));
        lock.unlock();

        LoadOutcome outcome;
        std::chrono::steady_clock::time_point load_completed_at;
        try {
            outcome = perform_load(model);
            load_completed_at = std::chrono::steady_clock::now();
        } catch (const std::exception& error) {
            lock.lock();
            reserved_loading_bytes_ -= bytes;
            loading_.erase(skill);
            ++metrics_.prefetch_failures;
            result.status = PrefetchStatus::Failed;
            result.reason = error.what();
            result.completed_at = std::chrono::steady_clock::now();
            load_condition_.notify_all();
            log_event("PREFETCH FAIL", "skill=" + skill + " reason=" +
                                           result.reason);
            return result;
        }

        lock.lock();
        reserved_loading_bytes_ -= bytes;
        loading_.erase(skill);
        finish_load_locked(model, outcome, true, used_soft_budget);
        ++metrics_.prefetch_successes;
        metrics_.prefetch_load_ms += outcome.duration_ms;
        metrics_.prefetch_latencies_ms.push_back(outcome.duration_ms);
        result.status = PrefetchStatus::Success;
        result.reason = used_soft_budget ? "soft_budget" : "hard_budget";
        result.load_ms = outcome.duration_ms;
        result.rss_before = outcome.backend.rss_before;
        result.rss_after = outcome.backend.rss_after;
        result.completed_at = load_completed_at;
        load_condition_.notify_all();
        log_event("PREFETCH READY", "skill=" + skill + " state=WARM load=" +
                                        std::to_string(result.load_ms) +
                                        "ms budget=" +
                                        (used_soft_budget ? "soft" : "hard") +
                                        " rss=" +
                                        std::to_string(result.rss_after));
        return result;
    } catch (const std::exception& error) {
        std::lock_guard lock(mutex_);
        ++metrics_.prefetch_failures;
        result.status = PrefetchStatus::Failed;
        result.reason = error.what();
        result.completed_at = std::chrono::steady_clock::now();
        log_event("PREFETCH FAIL", "skill=" + skill + " reason=" +
                                       result.reason);
        return result;
    }
}

void ModelManager::record_prefetch_observation(double overlap_ms,
                                                double wait_ms) {
    std::lock_guard lock(mutex_);
    metrics_.prefetch_overlap_ms += overlap_ms;
    metrics_.prefetch_wait_ms += wait_ms;
}

RuntimeMetrics ModelManager::metrics() const {
    std::lock_guard lock(mutex_);
    return metrics_;
}

ModelState ModelManager::model_state(const std::string& skill) const {
    std::lock_guard lock(mutex_);
    return find_model_locked(skill)->state();
}

std::size_t ModelManager::memory_budget() const noexcept { return memory_budget_; }
std::size_t ModelManager::prefetch_reserve() const noexcept {
    return prefetch_reserve_;
}

std::shared_ptr<Model> ModelManager::find_model_locked(
    const std::string& skill) const {
    const auto found = models_.find(skill);
    if (found == models_.end()) {
        throw std::out_of_range("unknown skill model: '" + skill + "'");
    }
    return found->second;
}

std::size_t ModelManager::file_size_locked(const std::string& skill) const {
    return file_sizes_.at(skill);
}

void ModelManager::ensure_capacity_locked(std::size_t required,
                                          const std::string& incoming) {
    if (!try_make_capacity_locked(memory_budget_, required, incoming)) {
        throw std::runtime_error("memory budget cannot fit model '" + incoming +
                                 "'; all eviction candidates are HOT or PINNED");
    }
}

bool ModelManager::try_make_capacity_locked(std::size_t limit,
                                            std::size_t required,
                                            const std::string& incoming) {
    const auto total = [&] {
        return metrics_.loaded_model_bytes + reserved_loading_bytes_ + required;
    };
    if (total() <= limit) return true;

    std::size_t evictable = 0;
    for (const auto& candidate : cache_policy_->eviction_candidates()) {
        if (candidate == incoming || loading_.contains(candidate)) continue;
        const auto model = find_model_locked(candidate);
        if (model->is_loaded() && !model->is_pinned() &&
            model->active_users() == 0) {
            evictable += model->size();
        }
    }
    if (total() - std::min(total(), evictable) > limit) return false;

    for (const auto& candidate : cache_policy_->eviction_candidates()) {
        if (total() <= limit) break;
        if (candidate != incoming && !loading_.contains(candidate)) {
            (void)evict_locked(candidate, "memory_budget");
        }
    }
    return total() <= limit;
}

ModelManager::LoadOutcome ModelManager::perform_load(
    const std::shared_ptr<Model>& model) {
    const auto start = std::chrono::steady_clock::now();
    bool dummy_mapped = false;
    bool backend_loaded = false;
    try {
        if (!inference_backend_.owns_model_memory()) {
            model->map_dummy();
            dummy_mapped = true;
        }
        LoadOutcome outcome;
        outcome.backend = inference_backend_.load(model->skill(),
                                                   model->path().string());
        backend_loaded = true;
        outcome.duration_ms = elapsed_ms(start);
        return outcome;
    } catch (...) {
        if (backend_loaded) inference_backend_.unload(model->skill());
        if (dummy_mapped && !model->is_pinned()) model->unmap_dummy();
        throw;
    }
}

void ModelManager::finish_load_locked(const std::shared_ptr<Model>& model,
                                      const LoadOutcome& outcome,
                                      bool is_prefetch,
                                      bool used_soft_budget) {
    model->mark_loaded();
    const std::size_t bytes = model->size();
    metrics_.loaded_model_bytes += bytes;
    metrics_.peak_loaded_model_bytes = std::max(
        metrics_.peak_loaded_model_bytes, metrics_.loaded_model_bytes);
    if (used_soft_budget && metrics_.loaded_model_bytes > memory_budget_) {
        metrics_.soft_budget_peak_bytes = std::max(
            metrics_.soft_budget_peak_bytes, metrics_.loaded_model_bytes);
    }
    metrics_.estimated_resident_memory = metrics_.loaded_model_bytes;
    metrics_.total_bytes_read += bytes;
    const double model_load_ms = outcome.backend.model_load_ms > 0.0
                                     ? outcome.backend.model_load_ms
                                     : outcome.duration_ms;
    metrics_.load_latencies_ms.push_back(model_load_ms);
    metrics_.context_creation_latencies_ms.push_back(
        outcome.backend.context_creation_ms);
    cache_policy_->touch(model->skill());

    log_event("MODEL LOAD", model->skill() + " path=" +
                                model->path().string() + " size=" +
                                megabytes(bytes) + " load_time=" +
                                std::to_string(model_load_ms) +
                                "ms rss_before=" +
                                std::to_string(outcome.backend.rss_before));
    log_event("MODEL READY", model->skill() + " context_time=" +
                                 std::to_string(
                                     outcome.backend.context_creation_ms) +
                                 "ms rss_after=" +
                                 std::to_string(outcome.backend.rss_after) +
                                 (is_prefetch ? " prefetched=true" : ""));
}

bool ModelManager::evict_locked(const std::string& skill,
                                const std::string& reason) {
    const auto model = find_model_locked(skill);
    if (!model->is_loaded()) {
        cache_policy_->erase(skill);
        return false;
    }
    if (model->is_pinned() || model->active_users() > 0 ||
        loading_.contains(skill)) {
        return false;
    }

    const std::size_t bytes = model->size();
    const ModelUnloadResult unload_metrics =
        inference_backend_.unload(model->skill());
    if (!inference_backend_.owns_model_memory()) model->unmap_dummy();
    model->mark_unloaded();
    metrics_.loaded_model_bytes -= bytes;
    metrics_.estimated_resident_memory = metrics_.loaded_model_bytes;
    metrics_.unload_latencies_ms.push_back(unload_metrics.unload_ms);
    if (reason == "memory_budget") ++metrics_.evictions;
    cache_policy_->erase(skill);
    log_event("EVICT", skill + " reason=" + reason + " unload_time=" +
                           std::to_string(unload_metrics.unload_ms) +
                           "ms rss_before=" +
                           std::to_string(unload_metrics.rss_before) +
                           " rss_after=" +
                           std::to_string(unload_metrics.rss_after));
    return true;
}

}  // namespace slm
