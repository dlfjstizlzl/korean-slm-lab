#include "inference/llama_cpp_inference_backend.hpp"

#include "core/logging.hpp"
#include "memory/macos/process_memory.hpp"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace slm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::mutex global_backend_mutex;
std::size_t global_backend_users = 0;

void filtered_llama_log(enum ggml_log_level level, const char* message, void*) {
    static std::mutex log_mutex;
    static bool previous_was_visible = false;
    std::lock_guard lock(log_mutex);
    const std::string_view text = message == nullptr ? "" : message;
    const bool device_info =
        level == GGML_LOG_LEVEL_INFO &&
        (text.find("offloaded ") != std::string_view::npos ||
         text.find("using device ") != std::string_view::npos);
    const bool visible = level == GGML_LOG_LEVEL_WARN ||
                         level == GGML_LOG_LEVEL_ERROR || device_info ||
                         (level == GGML_LOG_LEVEL_CONT && previous_was_visible);
    if (visible) {
        std::cerr << "[LLAMA] " << text;
    }
    if (level != GGML_LOG_LEVEL_CONT) {
        previous_was_visible = visible;
    }
}

void acquire_llama_runtime() {
    std::lock_guard lock(global_backend_mutex);
    if (global_backend_users++ == 0) {
        llama_log_set(filtered_llama_log, nullptr);
        llama_backend_init();
    }
}

void release_llama_runtime() noexcept {
    std::lock_guard lock(global_backend_mutex);
    if (global_backend_users > 0 && --global_backend_users == 0) {
        llama_backend_free();
    }
}

struct ModelDeleter {
    void operator()(llama_model* model) const noexcept {
        if (model != nullptr) llama_model_free(model);
    }
};

struct ContextDeleter {
    void operator()(llama_context* context) const noexcept {
        if (context != nullptr) llama_free(context);
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept {
        if (sampler != nullptr) llama_sampler_free(sampler);
    }
};

class LlamaModelInstance {
public:
    LlamaModelInstance(const std::string& model_name, const std::string& path,
                       const GenerationOptions& options, ModelLoadResult& metrics)
        : model_name_(model_name), options_(options) {
        metrics.rss_before = process_resident_bytes();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options_.gpu_layers;
        std::array<ggml_backend_dev_t, 2> cpu_devices{};
        if (options_.gpu_layers == 0) {
            cpu_devices[0] =
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_devices[0] == nullptr) {
                throw std::runtime_error("llama.cpp CPU backend is unavailable");
            }
            model_params.devices = cpu_devices.data();
        }

        const auto model_start = Clock::now();
        model_.reset(llama_model_load_from_file(path.c_str(), model_params));
        metrics.model_load_ms = elapsed_ms(model_start);
        if (!model_) {
            throw std::runtime_error("llama.cpp failed to load GGUF: '" + path + "'");
        }

        if (const char* embedded =
                llama_model_chat_template(model_.get(), /* name */ nullptr);
            embedded != nullptr && *embedded != '\0') {
            chat_template_ = embedded;
        }

        const ggml_backend_dev_t gpu_device =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        const ggml_backend_dev_t integrated_gpu =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        const ggml_backend_dev_t selected_gpu =
            gpu_device != nullptr ? gpu_device : integrated_gpu;
        std::ostringstream device;
        device << "skill=" << model_name_
               << " requested_gpu_layers="
               << (options_.gpu_layers < 0
                       ? "auto"
                       : std::to_string(options_.gpu_layers))
               << " backend=";
        if (options_.gpu_layers == 0) {
            device << "CPU gpu_layers_offloaded=0";
        } else if (selected_gpu != nullptr) {
            const ggml_backend_reg_t registry =
                ggml_backend_dev_backend_reg(selected_gpu);
            const char* registry_name =
                registry == nullptr ? "GPU" : ggml_backend_reg_name(registry);
            const std::string_view backend_name = registry_name;
            device << (backend_name == "MTL" ? "Metal" : registry_name)
                   << " device=" << ggml_backend_dev_name(selected_gpu);
            if (backend_name == "MTL") {
                device << " backend_registry=MTL";
            }
            device
                   << " gpu_layers_offloaded=llama_info_log"
                   << " actual_offload_count_api=unavailable";
        } else {
            device << "CPU gpu_layers_offloaded=0 gpu_request_unavailable=true";
        }
        device << " model_repeating_layers=" << llama_model_n_layer(model_.get());
        log_event("DEVICE", device.str());

        std::ostringstream format;
        format << "skill=" << model_name_ << " chat_template="
               << (chat_template_.empty() ? "missing_raw_fallback" : "embedded");
        if (model_name_ == "summary" && chat_template_.empty()) {
            format << " summary_quality_warning=use_instruction_tuned_gguf";
        }
        log_event("MODEL", format.str());

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = options_.context_size;
        context_params.n_batch = options_.context_size;
        context_params.n_threads = options_.cpu_threads;
        context_params.n_threads_batch = options_.cpu_threads;
        context_params.offload_kqv = options_.gpu_layers != 0;
        context_params.op_offload = options_.gpu_layers != 0;
        context_params.no_perf = false;

        const auto context_start = Clock::now();
        context_.reset(llama_init_from_model(model_.get(), context_params));
        metrics.context_creation_ms = elapsed_ms(context_start);
        if (!context_) {
            throw std::runtime_error("llama.cpp failed to create a context for '" +
                                     path + "'");
        }

        llama_sampler_chain_params sampler_params =
            llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        sampler_.reset(llama_sampler_chain_init(sampler_params));
        if (!sampler_) {
            throw std::runtime_error("llama.cpp failed to create a sampler");
        }
        if (!options_.grammar.empty()) {
            const llama_vocab* vocab = llama_model_get_vocab(model_.get());
            llama_sampler* grammar = llama_sampler_init_grammar(
                vocab, options_.grammar.c_str(), "root");
            if (grammar == nullptr) {
                throw std::runtime_error("failed to initialize generation grammar");
            }
            llama_sampler_chain_add(sampler_.get(), grammar);
        }
        llama_sampler_chain_add(sampler_.get(),
                                llama_sampler_init_top_p(options_.top_p, 1));
        llama_sampler_chain_add(sampler_.get(),
                                llama_sampler_init_temp(options_.temperature));
        llama_sampler_chain_add(sampler_.get(),
                                llama_sampler_init_dist(options_.seed));
        metrics.rss_after = process_resident_bytes();
    }

    GenerationResult generate(const std::string& prompt) {
        std::lock_guard lock(mutex_);
        GenerationResult result;
        const auto total_start = Clock::now();
        const llama_vocab* vocab = llama_model_get_vocab(model_.get());
        if (vocab == nullptr) {
            throw std::runtime_error("loaded model has no vocabulary");
        }
        if (prompt.size() > static_cast<std::size_t>(
                                std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("prompt is too large to tokenize");
        }

        llama_memory_clear(llama_get_memory(context_.get()), false);
        llama_sampler_reset(sampler_.get());

        const std::string formatted_prompt = format_prompt(prompt);
        result.prompt_format = chat_template_.empty()
                                   ? (options_.system_prompt.empty()
                                          ? "raw"
                                          : "raw_with_system")
                                   : "chat_template";
        if (formatted_prompt.size() > static_cast<std::size_t>(
                                          std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(
                "formatted prompt is too large to tokenize");
        }

        const auto tokenize_start = Clock::now();
        const auto prompt_size =
            static_cast<std::int32_t>(formatted_prompt.size());
        const std::int32_t needed = llama_tokenize(
            vocab, formatted_prompt.c_str(), prompt_size, nullptr, 0, true, true);
        if (needed == std::numeric_limits<std::int32_t>::min()) {
            throw std::runtime_error("prompt token count overflow");
        }
        const std::int32_t token_count = needed < 0 ? -needed : needed;
        if (token_count <= 0) {
            throw std::runtime_error("prompt produced no tokens");
        }
        std::vector<llama_token> prompt_tokens(
            static_cast<std::size_t>(token_count));
        const std::int32_t written = llama_tokenize(
            vocab, formatted_prompt.c_str(), prompt_size, prompt_tokens.data(),
            token_count, true, true);
        if (written < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }
        prompt_tokens.resize(static_cast<std::size_t>(written));
        result.prompt_tokens = prompt_tokens.size();
        result.tokenization_ms = elapsed_ms(tokenize_start);

        const std::size_t required_context =
            prompt_tokens.size() + options_.max_tokens;
        if (required_context > llama_n_ctx(context_.get())) {
            throw std::invalid_argument(
                "prompt tokens plus max tokens exceed configured context size");
        }

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<std::int32_t>(prompt_tokens.size()));
        const auto prefill_start = Clock::now();
        if (llama_model_has_encoder(model_.get())) {
            if (llama_encode(context_.get(), batch) != 0) {
                throw std::runtime_error("llama_encode failed");
            }
            llama_token decoder_start =
                llama_model_decoder_start_token(model_.get());
            if (decoder_start == LLAMA_TOKEN_NULL) {
                decoder_start = llama_vocab_bos(vocab);
            }
            batch = llama_batch_get_one(&decoder_start, 1);
        }
        if (llama_decode(context_.get(), batch) != 0) {
            throw std::runtime_error("llama_decode failed during prompt prefill");
        }
        result.prefill_ms = elapsed_ms(prefill_start);

        const auto decode_start = Clock::now();
        bool recorded_ttft = false;
        while (result.generated_tokens < options_.max_tokens) {
            const llama_token token =
                llama_sampler_sample(sampler_.get(), context_.get(), -1);
            if (!recorded_ttft) {
                result.ttft_ms = elapsed_ms(prefill_start);
                recorded_ttft = true;
            }
            if (llama_vocab_is_eog(vocab, token)) {
                result.stop_reason = GenerationStopReason::Eog;
                break;
            }

            result.text += token_piece(vocab, token);
            ++result.generated_tokens;
            if (result.generated_tokens >= options_.max_tokens) {
                break;
            }

            llama_token next = token;
            batch = llama_batch_get_one(&next, 1);
            if (llama_decode(context_.get(), batch) != 0) {
                throw std::runtime_error("llama_decode failed during generation");
            }
        }

        result.decode_ms = elapsed_ms(decode_start);
        result.total_ms = elapsed_ms(total_start);
        if (result.decode_ms > 0.0) {
            result.decode_tokens_per_second =
                1000.0 * result.generated_tokens / result.decode_ms;
        }
        result.rss_after_inference = process_resident_bytes();
        return result;
    }

private:
    std::string format_prompt(const std::string& prompt) const {
        if (chat_template_.empty()) {
            if (options_.system_prompt.empty()) return prompt;
            return options_.system_prompt + "\n\nUser request:\n" + prompt;
        }

        std::vector<llama_chat_message> messages;
        if (!options_.system_prompt.empty()) {
            messages.push_back({"system", options_.system_prompt.c_str()});
        }
        messages.push_back({"user", prompt.c_str()});
        std::int32_t required = llama_chat_apply_template(
            chat_template_.c_str(), messages.data(), messages.size(), true,
            nullptr, 0);
        if (required < 0) {
            throw std::runtime_error(
                "failed to size embedded chat template output for '" +
                model_name_ + "'");
        }
        std::vector<char> formatted(static_cast<std::size_t>(required) + 1);
        const std::int32_t written = llama_chat_apply_template(
            chat_template_.c_str(), messages.data(), messages.size(), true,
            formatted.data(), static_cast<std::int32_t>(formatted.size()));
        if (written < 0 || written > required) {
            throw std::runtime_error(
                "failed to apply embedded chat template for '" + model_name_ +
                "'");
        }
        return {formatted.data(), static_cast<std::size_t>(written)};
    }

    static std::string token_piece(const llama_vocab* vocab, llama_token token) {
        std::vector<char> buffer(256);
        std::int32_t length = llama_token_to_piece(
            vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()),
            0, true);
        if (length < 0) {
            buffer.resize(static_cast<std::size_t>(-length));
            length = llama_token_to_piece(
                vocab, token, buffer.data(),
                static_cast<std::int32_t>(buffer.size()), 0, true);
        }
        if (length < 0) {
            throw std::runtime_error("failed to convert generated token to text");
        }
        return {buffer.data(), static_cast<std::size_t>(length)};
    }

    std::string model_name_;
    GenerationOptions options_;
    std::string chat_template_;
    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> context_;
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler_;
    std::mutex mutex_;
};

std::string inference_log(const std::string& skill,
                          const GenerationResult& result) {
    std::ostringstream message;
    message << "skill=" << skill << " prompt_tokens=" << result.prompt_tokens
            << " generated_tokens=" << result.generated_tokens
            << " prompt_format=" << result.prompt_format
            << " stop_reason="
            << generation_stop_reason_name(result.stop_reason)
            << " tokenize=" << result.tokenization_ms << "ms"
            << " prefill=" << result.prefill_ms << "ms"
            << " ttft=" << result.ttft_ms << "ms"
            << " decode=" << result.decode_ms << "ms"
            << " total=" << result.total_ms << "ms"
            << " decode_speed=" << result.decode_tokens_per_second << " tok/s"
            << " rss_after=" << result.rss_after_inference;
    return message.str();
}

}  // namespace

class LlamaCppInferenceBackend::Impl {
public:
    explicit Impl(GenerationOptions generation_options)
        : options(std::move(generation_options)) {
        if (options.context_size == 0 || options.max_tokens == 0 ||
            options.cpu_threads <= 0 || options.temperature < 0.0F ||
            options.top_p <= 0.0F || options.top_p > 1.0F) {
            throw std::invalid_argument("invalid llama.cpp generation options");
        }
        acquire_llama_runtime();
    }

    ~Impl() {
        std::unordered_map<std::string, std::shared_ptr<LlamaModelInstance>>
            owned_models;
        {
            std::lock_guard lock(mutex);
            owned_models.swap(models);
        }
        owned_models.clear();
        release_llama_runtime();
    }

    GenerationOptions options;
    std::unordered_map<std::string, std::shared_ptr<LlamaModelInstance>> models;
    mutable std::mutex mutex;
};

LlamaCppInferenceBackend::LlamaCppInferenceBackend(GenerationOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

LlamaCppInferenceBackend::~LlamaCppInferenceBackend() = default;

ModelLoadResult LlamaCppInferenceBackend::load(const std::string& model_name,
                                                const std::string& model_path) {
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->models.contains(model_name)) return {};
    }

    ModelLoadResult metrics;
    auto instance = std::make_shared<LlamaModelInstance>(
        model_name, model_path, impl_->options, metrics);
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->models.contains(model_name)) {
            throw std::logic_error("duplicate concurrent llama.cpp load: '" +
                                   model_name + "'");
        }
        impl_->models.emplace(model_name, std::move(instance));
    }
    return metrics;
}

ModelUnloadResult LlamaCppInferenceBackend::unload(
    const std::string& model_name) {
    ModelUnloadResult metrics;
    metrics.rss_before = process_resident_bytes();
    const auto start = Clock::now();
    std::shared_ptr<LlamaModelInstance> removed;
    {
        std::lock_guard lock(impl_->mutex);
        const auto found = impl_->models.find(model_name);
        if (found != impl_->models.end()) {
            removed = std::move(found->second);
            impl_->models.erase(found);
        }
    }
    removed.reset();
    metrics.unload_ms = elapsed_ms(start);
    metrics.rss_after = process_resident_bytes();
    return metrics;
}

bool LlamaCppInferenceBackend::is_loaded(const std::string& model_name) const {
    std::lock_guard lock(impl_->mutex);
    return impl_->models.contains(model_name);
}

GenerationResult LlamaCppInferenceBackend::run(const std::string& model_name,
                                                const std::string& prompt) {
    std::shared_ptr<LlamaModelInstance> instance;
    {
        std::lock_guard lock(impl_->mutex);
        const auto found = impl_->models.find(model_name);
        if (found == impl_->models.end()) {
            throw std::logic_error("llama.cpp model is not loaded: '" +
                                   model_name + "'");
        }
        instance = found->second;
    }
    try {
        auto result = instance->generate(prompt);
        log_event("INFER", inference_log(model_name, result));
        return result;
    } catch (const std::exception& error) {
        log_event("INFER", "skill=" + model_name +
                               " stop_reason=error message=" + error.what());
        throw;
    }
}

bool LlamaCppInferenceBackend::owns_model_memory() const noexcept { return true; }
const char* LlamaCppInferenceBackend::name() const noexcept { return "llama"; }

std::unique_ptr<InferenceBackend> make_llama_cpp_inference_backend(
    GenerationOptions options) {
    return std::make_unique<LlamaCppInferenceBackend>(std::move(options));
}

}  // namespace slm
