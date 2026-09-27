#pragma once

#include <cstddef>

namespace slm {

// macOS reports ru_maxrss in bytes. This is process peak RSS, not per-model RSS.
std::size_t process_peak_resident_bytes() noexcept;

// Current resident size reported by Mach task_info. Kept in the macOS utility
// layer so the runtime core remains OS-independent.
std::size_t process_resident_bytes() noexcept;

}  // namespace slm
