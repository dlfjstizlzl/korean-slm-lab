#include "inference/inference_backend.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace slm {
namespace {

class DummyInferenceBackend final : public InferenceBackend {
public:
    ModelLoadResult load(const std::string& model_name,
                         const std::string&) override {
        std::lock_guard lock(mutex_);
        loaded_.insert(model_name);
        return {};
    }

    ModelUnloadResult unload(const std::string& model_name) override {
        std::lock_guard lock(mutex_);
        loaded_.erase(model_name);
        return {};
    }

    bool is_loaded(const std::string& model_name) const override {
        std::lock_guard lock(mutex_);
        return loaded_.contains(model_name);
    }

    GenerationResult run(const std::string& model_name,
                         const std::string& prompt) override {
        {
            std::lock_guard lock(mutex_);
            if (!loaded_.contains(model_name)) {
                throw std::logic_error("dummy model is not loaded: '" + model_name + "'");
            }
        }
        const auto start = std::chrono::steady_clock::now();
        // A small delay makes request timing visible without pretending to infer.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        GenerationResult result;
        result.text = "[dummy:" + model_name + "] " + prompt;
        result.generated_tokens = 1;
        result.stop_reason = GenerationStopReason::Eog;
        result.decode_ms = elapsed;
        result.total_ms = elapsed;
        result.decode_tokens_per_second = 1000.0 / elapsed;
        return result;
    }

    bool owns_model_memory() const noexcept override { return false; }
    const char* name() const noexcept override { return "dummy"; }

private:
    mutable std::mutex mutex_;
    std::unordered_set<std::string> loaded_;
};

}  // namespace

std::unique_ptr<InferenceBackend> make_dummy_inference_backend() {
    return std::make_unique<DummyInferenceBackend>();
}

}  // namespace slm
