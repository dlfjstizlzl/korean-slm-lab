#pragma once

#include "inference/inference_backend.hpp"

#include <memory>
#include <string>

namespace slm {

class LlamaCppInferenceBackend final : public InferenceBackend {
public:
    explicit LlamaCppInferenceBackend(GenerationOptions options);
    ~LlamaCppInferenceBackend() override;

    LlamaCppInferenceBackend(const LlamaCppInferenceBackend&) = delete;
    LlamaCppInferenceBackend& operator=(const LlamaCppInferenceBackend&) = delete;

    ModelLoadResult load(const std::string& model_name,
                         const std::string& model_path) override;
    ModelUnloadResult unload(const std::string& model_name) override;
    bool is_loaded(const std::string& model_name) const override;
    GenerationResult run(const std::string& model_name,
                         const std::string& prompt) override;
    bool owns_model_memory() const noexcept override;
    const char* name() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<InferenceBackend> make_llama_cpp_inference_backend(
    GenerationOptions options);

}  // namespace slm

