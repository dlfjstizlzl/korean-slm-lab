#include "cache/lru_cache.hpp"
#include "core/model_manager.hpp"
#include "manager/manager_client.hpp"
#include "manager/skill_prompt_adapter.hpp"
#include "manager/task_executor.hpp"
#include "inference/llama_cpp_inference_backend.hpp"
#include "memory/macos/mac_memory_backend.hpp"
#include "memory/macos/process_memory.hpp"
#include "prefetch/prefetcher.hpp"

#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <thread>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void make_file(const std::filesystem::path& path, std::size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    output.seekp(static_cast<std::streamoff>(bytes - 1));
    output.put('\0');
}

class RecordingInferenceBackend final : public slm::InferenceBackend {
public:
    slm::ModelLoadResult load(const std::string& model_name,
                              const std::string&) override {
        loading.store(true);
        if (load_delay.count() > 0) std::this_thread::sleep_for(load_delay);
        std::lock_guard lock(mutex);
        loaded.insert(model_name);
        ++load_count[model_name];
        loading.store(false);
        return {};
    }

    slm::ModelUnloadResult unload(const std::string& model_name) override {
        std::lock_guard lock(mutex);
        loaded.erase(model_name);
        ++unload_count[model_name];
        return {};
    }

    bool is_loaded(const std::string& model_name) const override {
        std::lock_guard lock(mutex);
        return loaded.contains(model_name);
    }

    slm::GenerationResult run(const std::string& model_name,
                              const std::string& prompt) override {
        {
            std::lock_guard lock(mutex);
            if (!loaded.contains(model_name)) {
                throw std::logic_error("model not loaded");
            }
        }
        if (run_delay.count() > 0) std::this_thread::sleep_for(run_delay);
        slm::GenerationResult result;
        result.text = prompt;
        return result;
    }

    bool owns_model_memory() const noexcept override { return true; }
    const char* name() const noexcept override { return "recording"; }

    int loads(const std::string& skill) const {
        std::lock_guard lock(mutex);
        const auto found = load_count.find(skill);
        return found == load_count.end() ? 0 : found->second;
    }

    int unloads(const std::string& skill) const {
        std::lock_guard lock(mutex);
        const auto found = unload_count.find(skill);
        return found == unload_count.end() ? 0 : found->second;
    }

    std::chrono::milliseconds load_delay{0};
    std::chrono::milliseconds run_delay{0};
    std::atomic<bool> loading{false};
    mutable std::mutex mutex;
    std::unordered_set<std::string> loaded;
    std::unordered_map<std::string, int> load_count;
    std::unordered_map<std::string, int> unload_count;
};

class FailOnMapMemoryBackend final : public slm::MemoryBackend {
public:
    void* map_file(const std::string&, std::size_t&) override {
        ++map_calls;
        throw std::logic_error("GGUF ownership path must not call MemoryBackend");
    }
    void unmap(void*, std::size_t) noexcept override {}
    void prefetch(void*, std::size_t) override {}
    void evict_hint(void*, std::size_t) override {}

    int map_calls = 0;
};

}  // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
                           "slm-runtime-v01-tests";
    try {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        make_file(directory / "a.bin", 4096);
        make_file(directory / "b.bin", 4096);
        make_file(directory / "c.bin", 4096);
        make_file(directory / "invalid.gguf", 4096);

        auto backend = std::make_shared<slm::MacMemoryBackend>();
        auto dummy = slm::make_dummy_inference_backend();
        slm::ModelManager manager(8192, backend,
                                  std::make_unique<slm::LruCachePolicy>(),
                                  *dummy);
        manager.register_model("a", (directory / "a.bin").string());
        manager.register_model("b", (directory / "b.bin").string());
        manager.register_model("c", (directory / "c.bin").string());

        { auto handle = manager.acquire("a"); }
        { auto handle = manager.acquire("a"); }
        { auto handle = manager.acquire("b"); }
        { auto handle = manager.acquire("c"); }
        const auto cache_metrics = manager.metrics();
        require(cache_metrics.cache_hits == 1, "expected one cache hit");
        require(cache_metrics.cache_misses == 3, "expected three cache misses");
        require(cache_metrics.evictions == 1, "expected one LRU eviction");
        require(cache_metrics.peak_loaded_model_bytes <= 8192,
                "loaded model cost exceeded budget");

        auto prefetch_dummy = slm::make_dummy_inference_backend();
        slm::ModelManager prefetch_manager(
            8192, backend, std::make_unique<slm::LruCachePolicy>(),
            *prefetch_dummy);
        prefetch_manager.register_model("a", (directory / "a.bin").string());
        slm::Prefetcher prefetcher(prefetch_manager);
        auto future = prefetcher.prefetch("a");
        (void)future.get();
        require(prefetch_manager.model_state("a") == slm::ModelState::Warm,
                "prefetched model should be WARM");

        // Simulate llama.cpp ownership: ModelManager must call backend unload on
        // budget eviction, then load again when the evicted skill is requested.
        RecordingInferenceBackend recording;
        auto fail_on_map = std::make_shared<FailOnMapMemoryBackend>();
        slm::ModelManager lifecycle_manager(
            4096, fail_on_map, std::make_unique<slm::LruCachePolicy>(), recording);
        lifecycle_manager.register_model("a", (directory / "a.bin").string());
        lifecycle_manager.register_model("b", (directory / "b.bin").string());
        { auto handle = lifecycle_manager.acquire("a"); }
        { auto handle = lifecycle_manager.acquire("b"); }
        { auto handle = lifecycle_manager.acquire("a"); }
        require(recording.loads("a") == 2,
                "evicted skill should be loaded again");
        require(recording.loads("b") == 1,
                "second skill should be loaded once");
        require(recording.unloads("a") == 1,
                "first skill should be unloaded by LRU");
        require(recording.unloads("b") == 1,
                "second skill should be unloaded before reloading first");
        require(recording.is_loaded("a"), "reloaded skill should be resident");
        lifecycle_manager.evict("a");
        require(!recording.is_loaded("a"),
                "manual eviction should unload backend object");
        require(fail_on_map->map_calls == 0,
                "backend-owned GGUF path must not create a second mapping");

        // TaskPlan v2 parser/validator/grammar/executor coverage lives in
        // tests/taskplan_v2_tests.cpp. Here a v2 plan drives the model
        // lifecycle and prefetch tests below; only Model-backed skills
        // (summary / extract / compare) load GGUF-style models.
        const slm::TaskPlan execution_plan = slm::TaskPlan::ok(
            {slm::Task{0, "summary", "core summary", {slm::SourceRef::user()}},
             slm::Task{1, "extract", "names only", {slm::SourceRef::task(0)}}},
            {1});
        const slm::RequestContext request_context{"virtual memory text"};
        slm::Runtime executor_runtime(
            8192, backend, std::make_unique<slm::LruCachePolicy>(),
            slm::make_dummy_inference_backend());
        executor_runtime.register_model("summary", (directory / "a.bin").string());
        executor_runtime.register_model("extract", (directory / "b.bin").string());
        slm::TaskExecutor executor(executor_runtime);
        const auto execution = executor.execute(execution_plan, request_context);
        require(execution.tasks.size() == 2,
                "executor should run every planned task");
        require(execution.tasks[1].skill_input.find(execution.tasks[0].output) !=
                    std::string::npos,
                "executor should pass a task source's output to the next task");
        require(execution.final_outputs.size() == 1 &&
                    execution.final_outputs[0].task_id == 1,
                "executor should return the tasks named in outputs");

        // v0.4 Test 1: a distinct next skill creates a plan-aware prefetch job
        // and its load overlaps the current skill's inference.
        auto plan_backend = std::make_unique<RecordingInferenceBackend>();
        auto* plan_backend_view = plan_backend.get();
        plan_backend_view->load_delay = std::chrono::milliseconds(25);
        plan_backend_view->run_delay = std::chrono::milliseconds(60);
        slm::Runtime plan_runtime(
            8192, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(plan_backend));
        plan_runtime.register_model("summary", (directory / "a.bin").string());
        plan_runtime.register_model("extract", (directory / "b.bin").string());
        slm::TaskExecutor plan_executor(
            plan_runtime, slm::SkillPromptAdapter{}, slm::PrefetchMode::Plan);
        const auto prefetched_execution =
            plan_executor.execute(execution_plan, request_context);
        const auto plan_metrics = plan_runtime.benchmark().model_manager;
        require(plan_metrics.prefetch_requests == 1 &&
                    plan_metrics.prefetch_successes == 1,
                "next distinct skill should create one successful prefetch");
        require(prefetched_execution.prefetch_overlap_ms > 0.0,
                "prefetch load should overlap current inference");
        require(plan_backend_view->loads("extract") == 1,
                "plan prefetch must load the next model once");

        // v0.4 Test 2: the last task has no successor to prefetch.
        auto last_backend = std::make_unique<RecordingInferenceBackend>();
        slm::Runtime last_runtime(
            4096, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(last_backend));
        last_runtime.register_model("summary", (directory / "a.bin").string());
        slm::TaskExecutor last_executor(
            last_runtime, slm::SkillPromptAdapter{}, slm::PrefetchMode::Plan);
        (void)last_executor.execute(
            slm::TaskPlan::ok({slm::Task{0, "summary", "x", {slm::SourceRef::user()}}},
                              {0}),
            request_context);
        require(last_runtime.benchmark().model_manager.prefetch_requests == 0,
                "last task must not create a prefetch job");

        // v0.4 Test 3: adjacent uses of the same skill reuse the HOT/WARM model
        // without a redundant prefetch request.
        auto same_backend = std::make_unique<RecordingInferenceBackend>();
        auto* same_backend_view = same_backend.get();
        slm::Runtime same_runtime(
            4096, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(same_backend));
        same_runtime.register_model("summary", (directory / "a.bin").string());
        slm::TaskExecutor same_executor(
            same_runtime, slm::SkillPromptAdapter{}, slm::PrefetchMode::Plan);
        (void)same_executor.execute(
            slm::TaskPlan::ok(
                {slm::Task{0, "summary", "x", {slm::SourceRef::user()}},
                 slm::Task{1, "summary", "y", {slm::SourceRef::task(0)}}},
                {1}),
            request_context);
        require(same_runtime.benchmark().model_manager.prefetch_requests == 0,
                "same-skill successor must not be prefetched");
        require(same_backend_view->loads("summary") == 1,
                "same-skill sequence should load once");

        // v0.4 Tests 4 and 7: reserve exhaustion safely skips prefetch and the
        // currently HOT model is never evicted to make room.
        RecordingInferenceBackend protected_backend;
        slm::ModelManager protected_manager(
            4096, 0, fail_on_map, std::make_unique<slm::LruCachePolicy>(),
            protected_backend);
        protected_manager.register_model("search",
                                         (directory / "a.bin").string());
        protected_manager.register_model("summary",
                                         (directory / "b.bin").string());
        {
            auto hot = protected_manager.acquire("search");
            const auto skipped = protected_manager.prefetch("summary");
            require(skipped.status == slm::PrefetchStatus::Skipped &&
                        skipped.reason == "budget",
                    "prefetch beyond hard plus reserve should skip");
            require(protected_manager.model_state("search") ==
                        slm::ModelState::Hot,
                    "prefetch must preserve the current HOT model");
            require(protected_backend.unloads("search") == 0,
                    "HOT model must not be evicted for prefetch");
        }

        // v0.4 Test 5: acquire arriving during an in-flight prefetch waits for
        // and reuses that exact load.
        auto inflight_backend = std::make_unique<RecordingInferenceBackend>();
        auto* inflight_backend_view = inflight_backend.get();
        inflight_backend_view->load_delay = std::chrono::milliseconds(80);
        slm::Runtime inflight_runtime(
            8192, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(inflight_backend));
        inflight_runtime.register_model("summary",
                                        (directory / "b.bin").string());
        const auto inflight = inflight_runtime.prefetch("summary");
        const auto loading_deadline = std::chrono::steady_clock::now() +
                                      std::chrono::seconds(2);
        while (!inflight_backend_view->loading.load() &&
               std::chrono::steady_clock::now() < loading_deadline) {
            std::this_thread::yield();
        }
        require(inflight_backend_view->loading.load(),
                "test did not observe the in-flight prefetch load");
        (void)inflight_runtime.run("summary", "x");
        require(inflight.get().status == slm::PrefetchStatus::Success,
                "in-flight prefetch should complete successfully");
        require(inflight_backend_view->loads("summary") == 1,
                "acquire during prefetch must not duplicate the load");

        // v0.4 Test 6: completed prefetch is a normal WARM cache hit.
        auto hit_backend = std::make_unique<RecordingInferenceBackend>();
        auto* hit_backend_view = hit_backend.get();
        slm::Runtime hit_runtime(
            4096, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(hit_backend));
        hit_runtime.register_model("summary", (directory / "b.bin").string());
        require(hit_runtime.prefetch("summary").get().status ==
                    slm::PrefetchStatus::Success,
                "prefetch should load a cold model");
        (void)hit_runtime.run("summary", "x");
        const auto hit_metrics = hit_runtime.benchmark().model_manager;
        require(hit_backend_view->loads("summary") == 1 &&
                    hit_metrics.cache_hits == 1,
                "acquire after prefetch should be a cache hit without reload");

        // v0.4 Test 8: disabling prefetch preserves v0.3 sequential LRU
        // behavior even when a two-task plan is known in advance.
        auto off_backend = std::make_unique<RecordingInferenceBackend>();
        slm::Runtime off_runtime(
            4096, 0, backend, std::make_unique<slm::LruCachePolicy>(),
            std::move(off_backend));
        off_runtime.register_model("summary", (directory / "a.bin").string());
        off_runtime.register_model("extract", (directory / "b.bin").string());
        slm::TaskExecutor off_executor(
            off_runtime, slm::SkillPromptAdapter{}, slm::PrefetchMode::Off);
        (void)off_executor.execute(execution_plan, request_context);
        const auto off_metrics = off_runtime.benchmark().model_manager;
        require(off_metrics.prefetch_requests == 0 &&
                    off_metrics.evictions == 1,
                "prefetch off should retain sequential v0.3 LRU switching");

        const slm::SkillPromptAdapter prompt_adapter;
        require(prompt_adapter.build_prompt("summary", "x", {{"user", "t"}})
                        .find("Summarize") != std::string::npos,
                "summary adapter contract is missing");

        require(slm::process_resident_bytes() > 0,
                "current process RSS should be observable on macOS");
        require(std::string(slm::generation_stop_reason_name(
                    slm::GenerationStopReason::Eog)) == "eog",
                "EOG stop reason should be observable");
        require(std::string(slm::generation_stop_reason_name(
                    slm::GenerationStopReason::MaxTokens)) == "max_tokens",
                "max-token stop reason should be observable");
        auto llama_backend =
            slm::make_llama_cpp_inference_backend(slm::GenerationOptions{});
        bool rejected_invalid_gguf = false;
        try {
            llama_backend->load("invalid", (directory / "invalid.gguf").string());
        } catch (const std::exception&) {
            rejected_invalid_gguf = true;
        }
        require(rejected_invalid_gguf, "llama.cpp should reject invalid GGUF data");
        require(!llama_backend->is_loaded("invalid"),
                "failed load must not leave a model instance resident");

        std::filesystem::remove_all(directory);
        std::cout << "All runtime tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        std::filesystem::remove_all(directory);
        return 1;
    }
}
