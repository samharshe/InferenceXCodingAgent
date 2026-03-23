# ARCHITECTURE.md
# Purpose: A map of the codebase. Use this to find code for specific functionality
# and to understand interfaces between components.

---

## Directory Tree (Annotated)

```
InferenceXCodingAgent/
│
├── .github/
│   ├── configs/
│   │   ├── nvidia-master.yaml       ← ALL NVIDIA benchmark definitions (models, frameworks, runners, search spaces)
│   │   ├── amd-master.yaml          ← ALL AMD benchmark definitions
│   │   ├── runners.yaml             ← Maps runner type (e.g. "b200") → actual runner node hostnames
│   │   └── CONFIGS.md               ← Human-readable docs for the YAML config format
│   └── workflows/
│       ├── run-sweep.yml            ← MAIN ENTRY POINT: parses changelog, generates matrix, fans out jobs
│       ├── benchmark-tmpl.yml       ← Single-node job template (called by run-sweep)
│       ├── benchmark-multinode-tmpl.yml ← Multi-node disagg job template (SLURM-based, AMD)
│       ├── collect-results.yml      ← Downloads artifacts → aggregates → uploads summary
│       ├── collect-evals.yml        ← Same for lm-eval results
│       ├── e2e-tests.yml            ← Manual trigger to test config generation
│       └── test-matrix-logic.yml    ← CI for unit tests on validation/generation
│
├── benchmarks/
│   ├── benchmark_lib.sh             ← Shared bash library: GPU monitor, server health, benchmark launch, eval run
│   ├── single_node/
│   │   └── {model}_{precision}_{gpu}.sh  ← One script per (model, precision, GPU) combo
│   │       Pattern: source lib → check env → download model → start GPU monitor →
│   │                launch server → wait health → run_benchmark_serving → optional eval
│   └── multi_node/
│       ├── {model}_{precision}_{gpu}_{framework}.sh  ← Multi-node SLURM scripts
│       └── amd_utils/
│           ├── submit.sh            ← SLURM job submission
│           ├── bench.sh             ← Multi-node benchmark runner
│           ├── server.sh            ← Server launch for multi-node
│           ├── job.slurm            ← SLURM job template
│           └── models.yaml          ← Model definitions for multi-node setup
│
├── utils/
│   ├── constants.py                 ← Master config file paths
│   ├── matrix_logic/
│   │   ├── validation.py            ← Pydantic models for ALL config shapes (input + output)
│   │   ├── generate_sweep_configs.py ← Expands master configs into matrix entries with filtering
│   │   ├── test_validation.py       ← Unit tests
│   │   └── test_generate_sweep_configs.py ← Unit tests
│   ├── bench_serving/
│   │   ├── benchmark_serving.py     ← THE BENCHMARK CLIENT: issues concurrent requests, measures TTFT/ITL/throughput
│   │   ├── backend_request_func.py  ← Backend-specific HTTP request logic (OpenAI, vLLM, SGLang, etc.)
│   │   └── benchmark_utils.py       ← Shared utilities
│   ├── process_changelog.py         ← Parses perf-changelog.yaml → which config keys to benchmark
│   ├── process_result.py            ← Extracts metrics from result JSONs
│   ├── summarize.py                 ← Generates markdown summary tables
│   ├── collect_results.py           ← Aggregates distributed result JSONs
│   └── evals/utils.py               ← lm-eval integration (GSM8K, GPQA Diamond)
│
├── experimental/
│   └── multiturn/
│       └── README.md               ← WIP lit review on multi-turn & KV cache offloading (not integrated)
│
├── perf-changelog.yaml             ← TRIGGER FILE: add entry here to trigger a benchmark run
├── README.md                       ← Project purpose + three new test descriptions
└── .claude/                        ← Claude Code work journals (this directory)
```

---

## Data Flow: Config to Results

```
Developer edits perf-changelog.yaml
            ↓
run-sweep.yml triggered (push to main)
            ↓
process_changelog.py → which config keys changed?
            ↓
generate_sweep_configs.py
  ← reads nvidia-master.yaml / amd-master.yaml
  ← validation.py validates input
  → expands: each config × seq-len × TP × EP × concurrency
  → validation.py validates output matrix entries
            ↓
GitHub Actions matrix (parallel jobs)
  each job → benchmark-tmpl.yml
           → runs benchmarks/single_node/{script}.sh
               → launches inference server (vLLM/SGLang/TRT)
               → calls run_benchmark_serving() in benchmark_lib.sh
                   → calls utils/bench_serving/benchmark_serving.py
                   → writes result JSON
            ↓
collect-results.yml aggregates all JSONs
            ↓
summarize.py → markdown tables + agg_bmk.json artifact
```

---

## Key Interfaces

### Master Config → Matrix Entry (validation.py)

**Input** (from `nvidia-master.yaml`):
```yaml
dsr1-fp4-b200-sglang:
  image: "..."
  model: "deepseek-ai/DeepSeek-R1"
  model-prefix: dsr1
  precision: fp4
  framework: sglang
  runner: b200
  seq-len-configs:
    - isl: 1024
      osl: 8192
      search-space:
        - tp: 1
          conc-start: 1
          conc-end: 256
```

**Output** (matrix entry fed to GitHub Actions):
```json
{
  "config_id": "dsr1-fp4-b200-sglang",
  "script": "dsr1_fp4_b200.sh",
  "tp": 1,
  "conc": 128,
  "isl": 1024,
  "osl": 8192,
  "runner": "b200-cw_0"
}
```

### benchmark_serving.py Key Arguments

```
--model              HF model ID
--backend            vllm | sglang | openai | tgi | etc.
--host / --port      Inference server address
--num-prompts        Total requests to issue
--request-rate       Requests/sec (inf = burst all)
--input-len          ISL (unique tokens per request)
--output-len         OSL
--prefix-len         Shared cached prefix length (added to input-len)
--save-result        Path to write JSON output
```

### benchmark_lib.sh: run_benchmark_serving()

Called from single-node scripts with env vars:
```bash
ISL=1024 OSL=8192 CONC=128 TP=1 run_benchmark_serving
```
Internally calls `benchmark_serving.py` with appropriate flags.

---

## Where to Add New Tests

For the three new multi-turn agentic tests:

1. **New orchestration script** in `benchmarks/single_node/` (or a new `multi_turn/` subdir)
   - Calls `benchmark_serving.py` in a loop with increasing ISL
   - Optionally inserts `sleep` between turns (Test 3)

2. **New seq-len-config type** in master YAML (e.g., `multiturn-sweep`) — or a separate config key

3. **New Pydantic models** in `validation.py` if the config shape differs materially

4. **New workflow** (or extension of `benchmark-tmpl.yml`) to handle the loop-based execution

5. **New result processing** in `process_result.py` / `summarize.py` for the turn-by-turn metric series
