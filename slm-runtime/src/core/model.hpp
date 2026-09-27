#pragma once

#include "memory/memory_backend.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace slm {

enum class ModelState {
    Cold,
    Mapped,
    Warm,
    Hot,
    Pinned,
};

const char* to_string(ModelState state) noexcept;

struct ModelDescriptor {
    std::string skill;
    std::filesystem::path path;
    std::size_t file_size = 0;
};

class Model {
public:
    Model(ModelDescriptor descriptor, bool pinned,
          std::shared_ptr<MemoryBackend> memory_backend);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void map_dummy();
    void prefetch_dummy();
    void unmap_dummy();
    void mark_loaded();
    void mark_hot();
    void release();
    void mark_unloaded();

    const std::string& skill() const noexcept;
    const std::filesystem::path& path() const noexcept;
    const ModelDescriptor& descriptor() const noexcept;
    std::size_t size() const noexcept;
    ModelState state() const noexcept;
    std::size_t active_users() const noexcept;
    bool is_pinned() const noexcept;
    bool is_loaded() const noexcept;
    bool is_mapped() const noexcept;

private:
    ModelDescriptor descriptor_;
    bool pinned_;
    std::shared_ptr<MemoryBackend> memory_backend_;
    void* address_ = nullptr;
    std::size_t mapped_size_ = 0;
    bool loaded_ = false;
    ModelState state_ = ModelState::Cold;
    std::size_t active_users_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace slm
