#include "core/model.hpp"

#include <stdexcept>
#include <utility>

namespace slm {

const char* to_string(ModelState state) noexcept {
    switch (state) {
        case ModelState::Cold: return "COLD";
        case ModelState::Mapped: return "MAPPED";
        case ModelState::Warm: return "WARM";
        case ModelState::Hot: return "HOT";
        case ModelState::Pinned: return "PINNED";
    }
    return "UNKNOWN";
}

Model::Model(ModelDescriptor descriptor, bool pinned,
             std::shared_ptr<MemoryBackend> memory_backend)
    : descriptor_(std::move(descriptor)),
      pinned_(pinned),
      memory_backend_(std::move(memory_backend)) {
    if (!memory_backend_) {
        throw std::invalid_argument("Model requires a memory backend");
    }
}

Model::~Model() {
    std::lock_guard lock(mutex_);
    memory_backend_->unmap(address_, mapped_size_);
}

void Model::map_dummy() {
    std::lock_guard lock(mutex_);
    if (address_ != nullptr) {
        return;
    }
    address_ = memory_backend_->map_file(descriptor_.path.string(), mapped_size_);
    if (mapped_size_ != descriptor_.file_size) {
        memory_backend_->unmap(address_, mapped_size_);
        address_ = nullptr;
        mapped_size_ = 0;
        throw std::runtime_error("model size changed while mapping: '" +
                                 descriptor_.skill + "'");
    }
    state_ = pinned_ ? ModelState::Pinned : ModelState::Mapped;
}

void Model::prefetch_dummy() {
    std::lock_guard lock(mutex_);
    if (address_ == nullptr) {
        throw std::logic_error("cannot prefetch an unmapped model");
    }
    memory_backend_->prefetch(address_, mapped_size_);
}

void Model::unmap_dummy() {
    std::lock_guard lock(mutex_);
    if (address_ == nullptr) return;
    if (pinned_ || active_users_ > 0) {
        throw std::logic_error("cannot unmap pinned or active model '" +
                               descriptor_.skill + "'");
    }
    memory_backend_->evict_hint(address_, mapped_size_);
    memory_backend_->unmap(address_, mapped_size_);
    address_ = nullptr;
    mapped_size_ = 0;
}

void Model::mark_loaded() {
    std::lock_guard lock(mutex_);
    loaded_ = true;
    state_ = pinned_ ? ModelState::Pinned : ModelState::Warm;
}

void Model::mark_hot() {
    std::lock_guard lock(mutex_);
    if (!loaded_) {
        throw std::logic_error("cannot acquire an unloaded model");
    }
    ++active_users_;
    if (!pinned_) {
        state_ = ModelState::Hot;
    }
}

void Model::release() {
    std::lock_guard lock(mutex_);
    if (active_users_ == 0) {
        return;
    }
    --active_users_;
    if (!pinned_ && active_users_ == 0) {
        state_ = ModelState::Warm;
    }
}

void Model::mark_unloaded() {
    std::lock_guard lock(mutex_);
    if (pinned_ || active_users_ > 0) {
        throw std::logic_error("cannot evict pinned or active model '" +
                               descriptor_.skill + "'");
    }
    loaded_ = false;
    state_ = ModelState::Cold;
}

const std::string& Model::skill() const noexcept { return descriptor_.skill; }
const std::filesystem::path& Model::path() const noexcept { return descriptor_.path; }
const ModelDescriptor& Model::descriptor() const noexcept { return descriptor_; }

std::size_t Model::size() const noexcept {
    return descriptor_.file_size;
}

ModelState Model::state() const noexcept {
    std::lock_guard lock(mutex_);
    return state_;
}

std::size_t Model::active_users() const noexcept {
    std::lock_guard lock(mutex_);
    return active_users_;
}

bool Model::is_pinned() const noexcept { return pinned_; }

bool Model::is_loaded() const noexcept {
    std::lock_guard lock(mutex_);
    return loaded_;
}

bool Model::is_mapped() const noexcept {
    std::lock_guard lock(mutex_);
    return address_ != nullptr;
}

}  // namespace slm
