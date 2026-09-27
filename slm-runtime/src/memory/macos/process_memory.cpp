#include "memory/macos/process_memory.hpp"

#include <mach/mach.h>
#include <sys/resource.h>

namespace slm {

std::size_t process_peak_resident_bytes() noexcept {
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return static_cast<std::size_t>(usage.ru_maxrss);
}

std::size_t process_resident_bytes() noexcept {
    mach_task_basic_info_data_t info {};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t result = ::task_info(
        ::mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&info), &count);
    if (result != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<std::size_t>(info.resident_size);
}

}  // namespace slm
