#include "memory/macos/mac_memory_backend.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace slm {
namespace {

std::runtime_error system_error(const std::string& action, const std::string& path = {}) {
    std::string message = action;
    if (!path.empty()) {
        message += " '" + path + "'";
    }
    message += ": ";
    message += std::strerror(errno);
    return std::runtime_error(message);
}

}  // namespace

void* MacMemoryBackend::map_file(const std::string& path, std::size_t& size) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw system_error("open failed for", path);
    }

    struct stat metadata {};
    if (::fstat(fd, &metadata) == -1) {
        const int saved_errno = errno;
        ::close(fd);
        errno = saved_errno;
        throw system_error("fstat failed for", path);
    }
    if (metadata.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("model file is empty: '" + path + "'");
    }

    size = static_cast<std::size_t>(metadata.st_size);
    void* address = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int saved_errno = errno;
    ::close(fd);
    if (address == MAP_FAILED) {
        errno = saved_errno;
        throw system_error("mmap failed for", path);
    }
    return address;
}

void MacMemoryBackend::unmap(void* address, std::size_t size) noexcept {
    if (address != nullptr && size > 0) {
        ::munmap(address, size);
    }
}

void MacMemoryBackend::prefetch(void* address, std::size_t size) {
    if (address != nullptr && size > 0 && ::madvise(address, size, MADV_WILLNEED) == -1) {
        throw system_error("madvise(MADV_WILLNEED) failed");
    }
}

void MacMemoryBackend::evict_hint(void* address, std::size_t size) {
    if (address != nullptr && size > 0 && ::madvise(address, size, MADV_DONTNEED) == -1) {
        throw system_error("madvise(MADV_DONTNEED) failed");
    }
}

}  // namespace slm

