// Thin harness around slm-runtime's own TaskPlan parser and validator.
//
// It links ../slm-runtime/src/manager/task_plan.cpp unchanged so the Python
// Manager dataset validator can be checked against the real Runtime behavior
// instead of a second, possibly drifting, implementation.
//
// stdin : one case per line, the raw Manager output encoded as lowercase hex
// stdout: "OK <TaskPlan::to_json()>" or "ERR <exception message>" per line
#include "manager/task_plan.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

std::string decode_hex(const std::string& hex) {
    std::string bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t index = 0; index + 1 < hex.size(); index += 2) {
        bytes.push_back(static_cast<char>(std::stoi(hex.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

}  // namespace

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto plan = slm::TaskPlan::from_manager_output(decode_hex(line));
            plan.validate();
            std::cout << "OK " << plan.to_json() << '\n';
        } catch (const std::exception& error) {
            std::cout << "ERR " << error.what() << '\n';
        }
    }
    return 0;
}
