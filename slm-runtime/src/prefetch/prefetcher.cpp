#include "prefetch/prefetcher.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace slm {

Prefetcher::Prefetcher(ModelManager& model_manager)
    : model_manager_(model_manager), worker_(&Prefetcher::worker_loop, this) {}

Prefetcher::~Prefetcher() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

std::shared_future<PrefetchResult> Prefetcher::prefetch(std::string skill) {
    std::lock_guard lock(mutex_);
    if (stopping_) {
        throw std::runtime_error("prefetcher is stopping");
    }
    if (const auto found = in_flight_.find(skill);
        found != in_flight_.end()) {
        return found->second;
    }

    auto job = std::make_unique<Job>();
    job->skill = std::move(skill);
    job->completion = std::make_shared<std::promise<PrefetchResult>>();
    job->future = job->completion->get_future().share();
    auto result = job->future;
    in_flight_.emplace(job->skill, result);
    jobs_.push(std::move(job));
    condition_.notify_one();
    return result;
}

void Prefetcher::worker_loop() {
    while (true) {
        std::unique_ptr<Job> job;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
            if (stopping_ && jobs_.empty()) {
                return;
            }
            job = std::move(jobs_.front());
            jobs_.pop();
        }

        try {
            job->completion->set_value(model_manager_.prefetch(job->skill));
        } catch (const std::exception& error) {
            PrefetchResult failed;
            failed.skill = job->skill;
            failed.status = PrefetchStatus::Failed;
            failed.reason = error.what();
            failed.started_at = std::chrono::steady_clock::now();
            failed.completed_at = failed.started_at;
            job->completion->set_value(std::move(failed));
        }
        {
            std::lock_guard lock(mutex_);
            const auto found = in_flight_.find(job->skill);
            if (found != in_flight_.end()) {
                in_flight_.erase(found);
            }
        }
    }
}

}  // namespace slm
