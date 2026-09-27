#pragma once

#include "memory/memory_backend.hpp"

namespace slm {

class MacMemoryBackend final : public MemoryBackend {
public:
    void* map_file(const std::string& path, std::size_t& size) override;
    void unmap(void* address, std::size_t size) noexcept override;
    void prefetch(void* address, std::size_t size) override;
    void evict_hint(void* address, std::size_t size) override;
};

}  // namespace slm

