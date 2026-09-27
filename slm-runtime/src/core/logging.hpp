#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace slm {

inline void log_event(std::string_view event, const std::string& message) {
    static std::mutex output_mutex;
    std::lock_guard lock(output_mutex);
    std::cout << '[' << event << ']';
    if (event.size() < 12) {
        std::cout << std::string(12 - event.size(), ' ');
    } else {
        std::cout << ' ';
    }
    std::cout << message << '\n';
}

}  // namespace slm
