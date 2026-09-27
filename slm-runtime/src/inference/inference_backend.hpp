#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace slm {

struct GenerationOptions {
    std::uint32_t context_size = 2048;
    std::uint32_t max_tokens = 128;
    float temperature = 0.7F;
    float top_p = 0.9F;
    std::uint32_t seed = 42;
    int gpu_layers = 0;
    int cpu_threads = 4;
    std::string system_prompt;
    std::string grammar;
};

struct ModelLoadResult {
    double model_load_ms = 0.0;
    double context_creation_ms = 0.0;
    std::size_t rss_before = 0;
    std::size_t rss_after = 0;
};

struct ModelUnloadResult {
    double unload_ms = 0.0;
    std::size_t rss_before = 0;
    std::size_t rss_after = 0;
};

enum class GenerationStopReason {
    Eog,
    MaxTokens,
    Error,
};

inline const char* generation_stop_reason_name(
    GenerationStopReason reason) noexcept {
    switch (reason) {
        case GenerationStopReason::Eog:
            return "eog";
        case GenerationStopReason::MaxTokens:
            return "max_tokens";
        case GenerationStopReason::Error:
            return "error";
    }
    return "error";
}

struct GenerationResult {
    std::string text;
    std::string prompt_format = "raw";
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    GenerationStopReason stop_reason = GenerationStopReason::MaxTokens;
    double tokenization_ms = 0.0;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    std::size_t rss_after_inference = 0;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual ModelLoadResult load(const std::string& model_name,
                                 const std::string& model_path) = 0;
    virtual ModelUnloadResult unload(const std::string& model_name) = 0;
    virtual bool is_loaded(const std::string& model_name) const = 0;
    virtual GenerationResult run(const std::string& model_name,
                                 const std::string& prompt) = 0;

    // True means this backend, rather than MemoryBackend, owns model mappings.
    virtual bool owns_model_memory() const noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

std::unique_ptr<InferenceBackend> make_dummy_inference_backend();

}  // namespace slm
