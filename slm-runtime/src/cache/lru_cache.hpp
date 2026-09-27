#pragma once

#include "cache/cache_policy.hpp"

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace slm {

class LruCachePolicy final : public CachePolicy {
public:
    void touch(const std::string& skill) override;
    void erase(const std::string& skill) override;
    std::vector<std::string> eviction_candidates() const override;

private:
    // Front is most recently used; back is least recently used.
    std::list<std::string> order_;
    std::unordered_map<std::string, std::list<std::string>::iterator> positions_;
    mutable std::mutex mutex_;
};

}  // namespace slm

