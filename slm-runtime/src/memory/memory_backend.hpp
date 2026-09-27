#pragma once

#include <cstddef>
#include <string>

namespace slm {

class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;

    virtual void* map_file(const std::string& path, std::size_t& size) = 0;
    virtual void unmap(void* address, std::size_t size) noexcept = 0;
    virtual void prefetch(void* address, std::size_t size) = 0;
    virtual void evict_hint(void* address, std::size_t size) = 0;
};

}  // namespace slm

