# SLM Runtime v0.5

Apple Silicon에서 여러 task-specific GGUF 모델을 필요할 때만
`llama.cpp`로 load하고, RAM budget 압력이 생기면 LRU 순서로 실제
`llama_model`, `llama_context`, `llama_sampler`를 해제하는 C++20
프로토타입이다.

v0.5는 Manager v1 / **TaskPlan v2**를 구현한다. PINNED Manager GGUF가 자연어
요청을 검증된 TaskPlan v2로 변환하고, 5개 Skill(`search`, `summary`,
`extract`, `compare`, `sql`)을 source 참조와 `outputs`에 따라 실행한다.
v0.4의 2-skill / `input`·`input_from` / 마지막 task만 반환하는 구조는
제거되었다(Pilot v1 재현용 snapshot은
`slm-korean-tokenizer/model/legacy_v1/runtime_v0_4_snapshot`).

```text
natural-language request
    -> pinned Manager SLM (TaskPlan v2 grammar)
    -> JSON parse + TaskPlan::validate (SkillRegistry)
    -> TaskExecutor: source resolution ("user" / earlier task ids)
         model skills  (summary / extract / compare) -> SkillPromptAdapter -> Runtime/ModelManager
         search        -> SearchBackend (retrieval)
         sql           -> SqlBackend (+ runtime-owned schema context)
    -> final outputs selected by `outputs`
```

Manager는 계획만 만든다. C++ `ModelManager`는 모델 load/cache/LRU/unload만
담당한다.

## Manager v1 skill catalog

`src/manager/skill_registry.cpp`의 `SkillRegistry::manager_v1()`이 single
source of truth다. Grammar, system prompt, validator, prompt adapter, executor
dispatch, main.cpp 모델 등록이 모두 이것에서 파생된다.

| Skill | 의미 | backend | sources |
|---|---|---|---|
| `search` | 외부 정보 검색(retrieval). instruction이 검색 의미 | Retrieval (`SearchBackend`) | 0 |
| `summary` | source 내용 요약 | Model (GGUF) | ≥ 1 |
| `extract` | instruction이 지정한 정보만 추출 | Model (GGUF) | ≥ 1 |
| `compare` | 대상/결과 비교. 한 문서 안 비교는 source 1개 허용 | Model (GGUF) | ≥ 1 |
| `sql` | 자연어 DB 요청. SQL 생성/실행은 SQL skill 책임 | Database (`SqlBackend`) | 0 |

**실제 검색 provider와 DB backend는 이 repository에 연결되어 있지 않다.**
기본값 `--search-backend none` / `--sql-backend none`은 해당 task를 명확한
오류로 실패시키고, `mock`은 결정적 offline test double이다(결과에 `MOCK` 표기).
DB schema/context는 `--sql-schema`로 runtime이 SQL backend에만 전달하며
TaskPlan에 복사되지 않는다.

## TaskPlan v2

```json
{"status":"ok","tasks":[{"id":0,"skill":"search","instruction":"Qwen3 최신 정보 검색","sources":[]},
 {"id":1,"skill":"summary","instruction":"주요 변경점 요약","sources":[0]}],"outputs":[1]}
{"status":"clarify","message":"요약할 내용을 제공해 주세요.","tasks":[],"outputs":[]}
{"status":"unsupported","message":"현재 지원하는 Skill로는 이 작업을 수행할 수 없습니다.","tasks":[],"outputs":[]}
```

- `status`: `ok` (1–8 tasks, outputs ≥ 1, message 없음) / `clarify` / `unsupported`
  (tasks·outputs 비어 있음, message 필수)
- task 필드는 `id`, `skill`, `instruction`, `sources`만 허용. `id` = 배열 index
- `sources`: `"user"`(사용자 원문/context) 또는 앞선 task id. self/future/중복 금지,
  skill별 source 개수 규칙 적용
- `outputs`: 사용자에게 반환할 task id (존재·중복 검사). 여러 개면 CLI는
  `[output i/n · task k · skill]` 형식으로 결정적으로 합친다
- 오류는 `TaskPlanError`의 안정적인 code(`INVALID_SKILL`, `FUTURE_REFERENCE`,
  `SOURCE_COUNT`, …)로 보고된다
- grammar(`ManagerClient::build_json_grammar`)는 registry에서 생성되며 canonical
  compact 형태만 허용한다(위치별 id, 이전 task만 참조, skill별 source 개수,
  범위 내 outputs)

긴 사용자 입력은 plan에 복사하지 않고 `"user"` source로 참조한다. Executor가
`RequestContext.user_content`(원 요청 + `--context-file` 내용)를 해당 skill에
전달한다.

## Manager SLM mode

```bash
./build/slm_runtime \
  --backend llama \
  --manager models/Qwen3-0.6B-Q4_0.gguf \
  --manager-budget-mb 700 \
  --skill summary=models/gemma-3-270m-Q8_0.gguf \
  --skill extract=models/gemma-3-270m-Q8_0.gguf \
  --skill compare=models/gemma-3-270m-Q8_0.gguf \
  --search-backend mock --sql-backend mock \
  --budget-mb 700 --gpu-layers 0 \
  --prompt "Qwen3 최신 정보를 찾아 요약해줘"
```

Manager generation은 temperature 0, 기본 최대 256 token, context ≥ 4096,
system prompt(12 few-shot) + TaskPlan v2 grammar를 사용한다. 검증 실패 시
correction prompt로 정확히 한 번 재시도한다. 현재 Manager GGUF는 TaskPlan v2로
학습되지 않았으므로 grammar-valid plan은 나오지만 의미는 맞지 않는다
(`slm-korean-tokenizer/outputs/manager_dataset_v2/real_manager_grammar_smoke.txt`).

`slm_taskplan_tool`(`--contract`, `--validate`, `--grammar`)은 registry, prompt,
grammar를 도구에 노출하고, llama.cpp grammar engine으로 plan 수용 여부를
판정한다. Python dataset validator가 이 도구로 differential test를 한다.

## Plan-Aware Prefetch

`--prefetch off`는 v0.3의 순차 load behavior를 유지하며 기본값이다.
`--prefetch plan`은 task `i`의 model을 HOT으로 acquire한 직후, 서로 다른
다음 skill이 있으면 background worker에 prefetch를 제출한 다음 현재
inference를 시작한다. 마지막 task, 같은 skill의 연속 실행, 이미 loaded된
model에는 새 load를 만들지 않는다.

```text
search acquire -> HOT
    |-- search inference -------------------|
    `-- summary prefetch -> WARM ------------|
search release -> WARM
summary acquire -> HOT
```

in-flight job은 skill별 shared future로 deduplicate한다. 다음 acquire는 진행
중인 동일 future를 기다리며 두 번째 load를 시작하지 않는다. Prefetch 실패나
budget skip은 요청 실패가 아니며 다음 task의 정상 acquire가 fallback이다.

llama backend registry는 짧은 mutex로만 보호한다. 각 loaded instance는
`shared_ptr` lifetime을 가지며 model별 context/sampler mutex를 사용한다.
따라서 서로 다른 model의 load와 inference는 겹칠 수 있고, unload는 현재
inference가 보유한 instance를 파괴하지 않는다. 전역 llama 초기화/해제의
reference count만 별도 mutex로 직렬화한다.

## Hard budget and prefetch reserve

```text
normal active accounting <= --budget-mb
prefetch-only temporary limit <= --budget-mb + --prefetch-reserve-mb
```

reserve 기본값은 0이다. Prefetch는 먼저 hard budget 안에서 WARM LRU candidate를
정리해 적재를 시도하고, 불가능하면 reserve가 허용하는 soft limit을 사용한다.
현재 HOT 또는 PINNED model은 prefetch를 위해 evict하지 않는다. Soft budget으로
적재된 WARM model을 실제 acquire할 때는 이전 HOT model이 release된 후 다른
WARM model을 LRU로 제거하여 다시 hard budget 안으로 복귀한다. Prefetch 완료
시 LRU에 등록하고 실제 acquire 시 recency를 갱신한다.

`peak_loaded_model_bytes`는 GGUF file-size 기반 논리 비용이고
`soft_budget_peak_bytes`는 hard budget을 넘긴 prefetch 시점의 전체 논리 비용
최고치다. 실제 process RSS에는 context/KV, allocator, page cache가 포함되므로
두 값은 별도로 보고한다.

## llama.cpp dependency

구현과 빌드에 사용한 공식 checkout:

```text
repository: https://github.com/ggml-org/llama.cpp
commit:     f172be756ae6e12a6aa294ecaeb2a387649ac6f8
version:    0.4.1-dev
```

이 공개용 저장소에는 `external/llama.cpp`가 포함되지 않는다. 최상위
README에 적힌 공식 저장소와 고정 커밋으로 별도 checkout한 뒤 빌드한다.
자세한 내용은 [external/README.md](external/README.md)에 있다.

CMake는 llama.cpp의 examples, tests, tools, server, app, UI를 끄고 `llama`
library만 추가한다. macOS에서는 CPU/Accelerate와 Metal을 함께 빌드한다.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

OpenMP가 없는 기본 Apple Clang에서는 configure warning이 발생할 수 있다.
CPU backend는 Accelerate를 사용하며 이 warning은 빌드 실패가 아니다.

AddressSanitizer build:

```bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSLM_ENABLE_ASAN=ON
cmake --build build-asan -j4
ctest --test-dir build-asan --output-on-failure
```

## Dummy backend

기존 v0.1 workload와 sparse model benchmark는 그대로 사용할 수 있다.

```bash
python3 tools/create_dummy_models.py
./build/slm_runtime --backend dummy
./build/slm_benchmark
```

## Connected local GGUF models

`models/`에 다음 파일이 있으면 `--backend llama` 실행 시 자동으로 역할에
연결한다. 따라서 현재 checkout에서는 `--model` 인자를 생략할 수 있다.

| Skill | Local GGUF |
|---|---|
| `search` | `models/Qwen3-0.6B-Q4_0.gguf` |
| `summary` | `models/gemma-3-270m-it-Q8_0.gguf`, 없으면 `gemma-3-270m-Q8_0.gguf` |
| `code` | `models/SmolLM3-Q4_K_M.gguf` |

현재 `gemma-3-270m-Q8_0.gguf` metadata에는 embedded chat template와
instruction-tuning 표식이 없다. 따라서 runtime은 raw fallback을 사용하고
요약 품질 경고를 출력한다. 요약 용도에는 `gemma-3-270m-it` 계열 GGUF를
`gemma-3-270m-it-Q8_0.gguf` 이름으로 배치하거나 `--model summary=...`로
지정하는 것을 권장한다. 모델은 자동으로 다운로드하지 않는다.

## Chat template and generation stop

모델에 `tokenizer.chat_template`가 있으면 `llama_model_chat_template()`로
읽고 `llama_chat_apply_template()`에 `user` message와 assistant generation
prompt를 전달한다. Template를 모델별로 하드코딩하지 않는다. Metadata에
template가 없을 때만 raw prompt로 fallback하며 로그에 이를 표시한다.

생성 token은 매번 `llama_vocab_is_eog()`로 확인한다. 종료 원인은
`stop_reason=eog`, `stop_reason=max_tokens`, 예외 시 `stop_reason=error`로
구분한다. `max_tokens`는 안전 상한이므로 chat template를 적용해도 긴
응답은 상한에서 끝날 수 있다.

자동 연결을 사용한 CPU-only 실행:

```bash
./build/slm_runtime \
  --backend llama \
  --gpu-layers 0 \
  --context 2048 \
  --max-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42 \
  --threads 4 \
  --prompt "Hello. Briefly explain what virtual memory is."
```

`--gpu-layers 0`은 model layer뿐 아니라 context KQV/op offload도 꺼서
correctness를 위한 CPU-only 경로로 실행한다.

다른 파일이나 역할을 사용하려면 자동 연결 대신 명시적으로 덮어쓴다.

```bash
./build/slm_runtime \
  --backend llama \
  --model search=/absolute/path/to/model.gguf \
  --gpu-layers 0
```

## Metal

현재 llama.cpp에서 음수 `n_gpu_layers`는 모든 layer를 offload한다. CLI의
`auto`는 이를 `-1`로 전달한다.

```bash
./build/slm_runtime \
  --backend llama \
  --model search=/absolute/path/to/model.gguf \
  --gpu-layers auto
```

실행 로그의 `[DEVICE]`는 요청값, backend registry와 device를 출력한다.
현재 public llama API는 실제 offload layer 수를 반환하지 않으므로 그 값을
추측하지 않는다. 대신 llama.cpp 자체 INFO 로그의
`offloaded N/M layers to GPU`를 선택적으로 노출해 실제 수를 확인한다.

정확한 layer 수를 직접 지정할 수도 있다.

```bash
./build/slm_runtime \
  --backend llama \
  --model search=/absolute/path/to/model.gguf \
  --gpu-layers 20
```

## Two skills and real unload/reload

동일한 GGUF를 두 skill에 등록해도 lifecycle을 시험할 수 있다. Budget은
모델 하나는 들어가지만 두 개는 동시에 들어가지 않는 값으로 지정한다.

```bash
./build/slm_runtime \
  --backend llama \
  --model search=models/Qwen3-0.6B-Q4_0.gguf \
  --model summary=models/gemma-3-270m-Q8_0.gguf \
  --budget-mb 500 \
  --gpu-layers 0 \
  --max-tokens 64
```

이 경우 다음 workload가 실행된다.

```text
search, search, summary, search, summary
```

각 요청의 prompt와 generated output이 stdout에 출력된다. 마지막에는 남아
있는 모델도 명시적으로 unload하여 unload 이후 RSS를 기록한다.

## CLI options

```text
--backend dummy|llama
--model skill=/absolute/path/model.gguf   repeatable
--skill skill=/absolute/path/model.gguf   repeatable, preferred
--manager /absolute/path/manager.gguf
--manager-budget-mb N                    default 1024, separate and pinned
--manager-max-tokens N                   default 256
--budget-mb N                            default: dummy 1200, llama 2048
--prefetch off|plan                      default off
--prefetch-reserve-mb N                  default 0
--context N                              default 2048
--max-tokens N                           default 128
--temperature F                          default 0.7
--top-p F                                default 0.9
--seed N                                 default 42
--threads N                              default 4
--gpu-layers N|auto                      default 0
--prompt TEXT
--interactive
```

## Metrics and logs

v0.4 separates:

- model load latency
- context creation latency
- prompt tokenization latency
- prompt prefill latency
- time to first token (TTFT)
- decode latency and decode tokens/sec (prompt tokens excluded)
- total inference latency
- unload latency
- current RSS before load, after load, after inference, and after unload
- process peak RSS
- cache hits/misses, budget evictions, and peak loaded-model cost
- manager load, TTFT, generated tokens, decode speed, and planning latency
- task count, skill switching, skill inference, and end-to-end request latency
- prefetch requests/hits/successes/skips/failures and prefetch load time
- timestamp-based prefetch overlap, acquire wait, hidden ratio, and visible switch
- hard-budget peak, soft-budget peak, and process peak RSS

Example format:

```text
[MODEL LOAD] search path=/models/model.gguf size=... load_time=... rss_before=...
[MODEL READY] search context_time=... rss_after=...
[DEVICE]     requested_gpu_layers=auto backend=Metal device=MTL0 ...
[MODEL]      skill=search chat_template=embedded
[INFER]      skill=search prompt_tokens=... generated_tokens=... prompt_format=chat_template stop_reason=eog tokenize=... prefill=... ttft=... decode=... total=... decode_speed=... tok/s
[MODEL HIT]  search state=WARM
[EVICT]      search reason=memory_budget unload_time=... rss_before=... rss_after=...
[OUTPUT]     generated text
[MANAGER PLAN] tasks=2 attempts=1
[TASK] id=1 skill=summary input_source=task:0
[PREFETCH START] skill=summary size=... budget=soft
[PREFETCH READY] skill=summary state=WARM load=... rss=...
[PREFETCH RESULT] skill=summary load=... overlap=... wait=... hidden_ratio=...
[REQUEST METRICS] manager_latency=... model_switch_latency=... prefetch_time=... prefetch_overlap=... prefetch_wait=... visible_switch_latency=... total_latency=...
```

Current RSS uses the macOS Mach `task_info` API in the OS-specific process-memory
utility. `MacMemoryBackend` itself remains POSIX-only. macOS may keep filesystem
pages in the system page cache after `llama_model_free`; page cache retention is
not the same as a live llama object or a memory leak.

TTFT는 tokenization이 끝난 뒤 prompt prefill을 시작한 시점부터 첫 token을
sample한 시점까지다. `decode_speed`는 generated token 수를 `decode` 시간으로
나눈 값이며 prompt token은 포함하지 않는다.

## Budget semantics

```text
v0.4 skill budget metric = GGUF logical file size approximation
```

Context/KV memory, Metal allocation behavior, shared filesystem cache, split GGUF
files, and exact resident pages are not part of the budget yet. The approximation
is deliberately conservative and simple for lifecycle experiments.

## Tests

The tests cover:

- dummy `mmap`, cache hit/miss, and LRU budget enforcement
- asynchronous prefetch and `WARM` state
- backend-owned load -> LRU unload -> reload lifecycle using a recording backend
- manual unload reaching the inference backend
- TaskPlan v2 (`tests/taskplan_v2_tests.cpp`, 145 checks): parse/serialize
  roundtrip, compact canonical output, status semantics, sources, multi-source,
  outputs, clarify/unsupported, self/future/duplicate references, skill-specific
  source counts, duplicate keys, float ids, max tasks, invalid UTF-8/U+FFFD,
  legacy `input`/`input_from` rejection
- generated grammar parsed and executed by llama.cpp's grammar engine: accepts
  every few-shot and canonical plan, rejects invalid structures
- executor source resolution (`user`, earlier task, multi-source compare),
  multiple final outputs, search/sql backend contracts, unconfigured backends
  failing loudly, no model prompt for search/sql
- 8-scenario orchestration smoke through ManagerClient + TaskExecutor with a
  scripted Manager and mock search/sql
- exactly one Manager correction retry and safe failure after the retry
- current macOS RSS measurement
- invalid GGUF rejection without leaving a loaded llama instance
- next-skill job creation, last-task skip, and same-skill deduplication
- hard plus reserve budget skip and HOT model protection
- in-flight acquire reuse without duplicate load and completed-prefetch cache hit
- prefetch-off v0.3 sequential LRU behavior

2026-09-17에 같은 prompt, context 1024, max tokens 128, temperature 0.7,
top-p 0.9, seed 42, threads 4로 Qwen3 CPU/Metal을 비교했다. 성능 보장값은
아니다.

| Qwen3 0.6B | CPU | Metal full offload |
|---|---:|---:|
| load | 223.23 ms | 98.93 ms |
| context | 7.90 ms | 18.40 ms |
| prefill | 21.06 ms | 75.48 ms |
| TTFT | 22.14 ms | 95.46 ms |
| decode | 263.78 tok/s | 293.52 tok/s |
| inference total | 506.75 ms | 511.76 ms |
| process peak RSS | 823.11 MiB | 645.58 MiB |

Metal 실행에서는 llama.cpp가 `offloaded 29/29 layers to GPU`를 보고했다.
동일한 64-token Gemma 실행은 CPU 272.57 tok/s, Metal 261.89 tok/s였고
Metal은 `19/19` layer offload를 보고했다. Gemma의 출력 품질은 두 backend
모두 낮았으며, backend가 아니라 base GGUF에 chat template가 없는 것이
원인이다.

500 MiB CPU LRU workload에서는 요청 5회, cache hit 1회, miss 4회,
budget eviction 3회를 기록했다. Qwen과 Gemma가 각각 두 번 load되었고
이전 모델은 매번 실제 unload되었다. 각 unload 후 RSS는 53.84, 90.13,
90.33, 91.44 MiB였고 최종 report는 91.44 MiB였다.

2026-09-18 v0.3 실제 CPU 실행에서는 Qwen3 Manager가 Test A를 `search`,
Test B를 `summary`, Test C를 `search -> summary`로 각각 한 번에 routing했다.
Test C raw plan은 `input_from: 0`을 포함했고 500 MiB skill budget에서 search
unload 후 summary load가 발생해 `Evictions: 1`을 기록했다.

| v0.3 request | Manager | Skill inference | Total | Tasks |
|---|---:|---:|---:|---:|
| Test A search | 827.78 ms | 274.51 ms | 1248.02 ms | 1 |
| Test B summary | 908.39 ms | 256.24 ms | 1410.21 ms | 1 |
| Test C search → summary | 1138.02 ms | 518.07 ms | 2047.08 ms | 2 |

Test C 종료 보고 시 skill은 unload됐지만 Manager 409.10 MiB는 PINNED 상태로
남았고 Manager eviction은 0이었다. 이 시점 process RSS는 1031.28 MiB였다.
Manager와 search가 같은 GGUF여도 현재는 별도 instance이므로 중복 resident
memory가 생긴다.

## v0.4 Plan-Aware Prefetch benchmark

2026-09-18 Apple M5 Pro CPU-only 환경에서 Qwen3 0.6B Manager/Search와
Gemma 3 270M Summary를 사용했다. Prompt, seed, 500 MiB hard skill budget,
64 skill tokens, 128 Manager tokens는 전 조건에서 동일했다. 각 조건을 같은
프로세스에서 11회 실행했고 첫 request는 process-cold로 분리한 뒤 나머지
10회만 steady-state 평균에 사용했다. OS filesystem cache까지 비운 disk-cold
측정은 아니다. Manager raw plan은 모두 동일한 `search -> summary`였다.

| Steady-state (n=10) | OFF | Plan, reserve 0 | Plan, reserve 100 | Plan, reserve 300 |
|---|---:|---:|---:|---:|
| Manager latency | 1140.10 ms | 1157.14 ms | 1183.57 ms | 1190.51 ms |
| Search inference | 274.41 ms | 274.10 ms | 279.08 ms | 292.45 ms |
| Summary load/acquire wait | 205.58 ms | 215.04 ms | 215.19 ms | 14.96 ms |
| Summary inference | 251.50 ms | 250.68 ms | 255.57 ms | 256.90 ms |
| Visible switch time | 361.77 ms | 370.50 ms | 374.91 ms | 175.25 ms |
| Total request | 2027.85 ms | 2052.50 ms | 2093.20 ms | 1915.19 ms |
| Process peak RSS | 1965.75 MiB | 1966.77 MiB | 1962.69 MiB | 2553.97 MiB |
| Peak skill cost | 409.10 MiB | 409.10 MiB | 409.10 MiB | 687.14 MiB |
| Soft-budget peak | 0 MiB | 0 MiB | 0 MiB | 687.14 MiB |
| Prefetch result | disabled | 11/11 skip | 11/11 skip | 11/11 success |

Reserve 300 MiB의 steady-state prefetch load 평균은 272.64 ms였고 timestamp
교집합으로 측정한 overlap도 272.64 ms였다. 평균 wait는 0.001 ms 미만으로
hidden ratio는 100%였다. OFF 대비 visible switch는 186.52 ms(51.56%), 전체
request는 112.65 ms(5.56%) 감소했다. 반면 process peak RSS는 588.22 MiB,
논리 skill cost peak는 278.04 MiB 증가했다. 동시 load의 CPU/memory-bandwidth
경합으로 Search inference는 18.04 ms 느려졌으므로 load overlap 전부가
end-to-end 절감으로 이어지지는 않았다.

첫 request의 process-cold 비교는 다음과 같다.

| Cold request | OFF | Plan, reserve 300 |
|---|---:|---:|
| Manager latency | 1134.41 ms | 1180.74 ms |
| Search inference | 280.65 ms | 285.25 ms |
| Summary load/acquire wait | 218.19 ms | 14.40 ms |
| Summary inference | 250.09 ms | 251.62 ms |
| Visible switch | 364.52 ms | 166.61 ms |
| Total request | 2029.74 ms | 1884.31 ms |
| Peak RSS at request | 1923.64 MiB | 2537.95 MiB |

## Scope limits (v0.5)

- no predictive prefetch
- no statistical prefetch or cancellation policy
- no partial GGUF/page/layer residency
- no Manager/skill model instance sharing
- no parallel task execution or multi-user concurrency
- no real external search provider or SQL database (interfaces + deterministic mocks only)
- no custom GGUF parser or llama.cpp mmap modification
- no custom page scheduler
- no pruning or fine-tuning
- no CPU/GPU automatic scheduler beyond explicit `--gpu-layers`
- no concurrent inference requests
- no Linux or Windows backend

## Next step

현재 Gemma base GGUF를 instruction-tuned GGUF로 교체한 뒤 요약 품질과 EOG
종료를 다시 측정하는 것이 우선이다. 다음 runtime 연구는 shared backbone이나
partial residency가 실제 RSS 증가를 줄일 수 있는지 검증하는 것이다.
