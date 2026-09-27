#include "cache/lru_cache.hpp"

namespace slm {

void LruCachePolicy::touch(const std::string& skill) {
    std::lock_guard lock(mutex_);
    if (const auto found = positions_.find(skill); found != positions_.end()) {
        order_.erase(found->second);
    }
    order_.push_front(skill);
    positions_[skill] = order_.begin();
}

void LruCachePolicy::erase(const std::string& skill) {
    std::lock_guard lock(mutex_);
    const auto found = positions_.find(skill);
    if (found == positions_.end()) {
        return;
    }
    order_.erase(found->second);
    positions_.erase(found);
}

std::vector<std::string> LruCachePolicy::eviction_candidates() const {
    std::lock_guard lock(mutex_);
    return {order_.rbegin(), order_.rend()};
}

}  // namespace slm

