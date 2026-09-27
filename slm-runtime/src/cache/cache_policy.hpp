#pragma once

#include <string>
#include <vector>

namespace slm {

class CachePolicy {
public:
    virtual ~CachePolicy() = default;

    virtual void touch(const std::string& skill) = 0;
    virtual void erase(const std::string& skill) = 0;
    virtual std::vector<std::string> eviction_candidates() const = 0;
};

}  // namespace slm

