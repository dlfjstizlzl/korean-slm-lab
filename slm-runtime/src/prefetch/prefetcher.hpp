#pragma once

#include "core/model_manager.hpp"

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

namespace slm {

class Prefetcher {
public:
    explicit Prefetcher(ModelManager& model_manager);
    ~Prefetcher();

    Prefetcher(const Prefetcher&) = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    std::shared_future<PrefetchResult> prefetch(std::string skill);

private:
    struct Job {
        std::string skill;
        std::shared_ptr<std::promise<PrefetchResult>> completion;
        std::shared_future<PrefetchResult> future;
    };

    void worker_loop();

    ModelManager& model_manager_;
    std::queue<std::unique_ptr<Job>> jobs_;
    std::unordered_map<std::string, std::shared_future<PrefetchResult>> in_flight_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace slm
