cmake_minimum_required(VERSION 3.20)

project(slm_runtime VERSION 0.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT APPLE)
    message(FATAL_ERROR "v0.4 currently provides only the macOS memory backend")
endif()

find_package(Threads REQUIRED)

option(SLM_ENABLE_ASAN "Enable AddressSanitizer for the runtime" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp is missing. Clone it into external/llama.cpp; see README.md")
endif()

# llama.cpp is a library dependency. Do not build its CLI tools, examples, or tests.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(LLAMA_SANITIZE_ADDRESS ${SLM_ENABLE_ASAN} CACHE BOOL "" FORCE)
add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)

add_library(slm_runtime_lib
    src/core/model.cpp
    src/core/model_manager.cpp
    src/core/runtime.cpp
    src/cache/lru_cache.cpp
    src/memory/macos/mac_memory_backend.cpp
    src/memory/macos/process_memory.cpp
    src/prefetch/prefetcher.cpp
    src/inference/dummy_inference_backend.cpp
    src/inference/llama_cpp_inference_backend.cpp
    src/manager/manager_client.cpp
    src/manager/task_plan.cpp
    src/manager/skill_prompt_adapter.cpp
    src/manager/task_executor.cpp
    src/manager/skill_registry.cpp
    src/skills/search_backend.cpp
    src/skills/sql_backend.cpp
)

target_include_directories(slm_runtime_lib PUBLIC src)
target_link_libraries(slm_runtime_lib PUBLIC Threads::Threads llama)
target_compile_options(slm_runtime_lib PRIVATE -Wall -Wextra -Wpedantic)

if(SLM_ENABLE_ASAN)
    target_compile_options(slm_runtime_lib PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(slm_runtime_lib PUBLIC -fsanitize=address)
endif()

add_executable(slm_runtime src/main.cpp)
target_link_libraries(slm_runtime PRIVATE slm_runtime_lib)

# TaskPlan v2 contract/validation/grammar tool used by the Python dataset
# validator for differential testing. It links llama.cpp's grammar engine to
# check grammar acceptance without a model vocabulary.
add_executable(slm_taskplan_tool tools/taskplan_tool.cpp)
target_link_libraries(slm_taskplan_tool PRIVATE slm_runtime_lib)
target_include_directories(slm_taskplan_tool PRIVATE external/llama.cpp/src)

add_executable(slm_benchmark benchmark/benchmark.cpp)
target_link_libraries(slm_benchmark PRIVATE slm_runtime_lib)

include(CTest)
if(BUILD_TESTING)
    add_executable(slm_runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(slm_runtime_tests PRIVATE slm_runtime_lib)
    add_test(NAME slm_runtime_tests COMMAND slm_runtime_tests)
    add_executable(slm_taskplan_v2_tests tests/taskplan_v2_tests.cpp)
    target_link_libraries(slm_taskplan_v2_tests PRIVATE slm_runtime_lib)
    target_include_directories(slm_taskplan_v2_tests PRIVATE external/llama.cpp/src)
    add_test(NAME slm_taskplan_v2_tests COMMAND slm_taskplan_v2_tests)
endif()
