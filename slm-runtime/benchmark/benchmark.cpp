#include "cache/lru_cache.hpp"
#include "core/runtime.hpp"
#include "inference/inference_backend.hpp"
#include "memory/macos/mac_memory_backend.hpp"
#include "memory/macos/process_memory.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr std::size_t mib = 1024ULL * 1024ULL;

double average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

std::filesystem::path default_model_directory() {
    if (std::filesystem::exists("../models/search.bin")) return "../models";
    return "models";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const auto model_directory = argc > 1
                                         ? std::filesystem::path(argv[1])
                                         : default_model_directory();
        const std::size_t budget_mb = argc > 2 ? std::stoull(argv[2]) : 1200;

        slm::Runtime runtime(budget_mb * mib,
                             std::make_shared<slm::MacMemoryBackend>(),
                             std::make_unique<slm::LruCachePolicy>(),
                             slm::make_dummy_inference_backend());
        for (const auto& skill : {"search", "sql", "summary", "code"}) {
            runtime.register_model(
                skill, (model_directory / (std::string(skill) + ".bin")).string());
        }

        auto prefetch = runtime.prefetch("summary");
        (void)prefetch.get();
        const std::vector<std::string> workload{
            "search", "search", "summary", "sql", "search",
            "code", "summary", "sql", "search",
        };
        for (const auto& skill : workload) {
            runtime.run(skill, "benchmark request");
        }

        const auto result = runtime.benchmark();
        const auto& metrics = result.model_manager;
        const std::size_t requests = metrics.cache_hits + metrics.cache_misses;
        const double hit_rate = requests == 0
                                    ? 0.0
                                    : 100.0 * metrics.cache_hits / requests;

        std::cout << "\n=== Runtime Benchmark ===\n\n"
                  << "Requests: " << requests << '\n'
                  << "Cache hits: " << metrics.cache_hits << '\n'
                  << "Cache misses: " << metrics.cache_misses << '\n'
                  << std::fixed << std::setprecision(1)
                  << "Hit rate: " << hit_rate << "%\n\n"
                  << std::setprecision(2)
                  << "Average model load latency: "
                  << average(metrics.load_latencies_ms) << " ms\n"
                  << "Average prefetch latency: "
                  << average(metrics.prefetch_latencies_ms) << " ms\n"
                  << "Average request latency: "
                  << average(result.request_latencies_ms) << " ms\n"
                  << "Average model switch latency: "
                  << average(result.model_switch_latencies_ms) << " ms\n\n"
                  << "Peak loaded-model cost: "
                  << static_cast<double>(metrics.peak_loaded_model_bytes) / (1024.0 * mib)
                  << " GB\n"
                  << "Estimated resident memory: "
                  << static_cast<double>(metrics.estimated_resident_memory) / mib
                  << " MB\n"
                  << "Process peak RSS (macOS): "
                  << static_cast<double>(slm::process_peak_resident_bytes()) / mib
                  << " MB\n"
                  << "Evictions: " << metrics.evictions << '\n'
                  << "Average unload latency: "
                  << average(metrics.unload_latencies_ms) << " ms\n"
                  << "Total bytes read (estimated): " << metrics.total_bytes_read
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
