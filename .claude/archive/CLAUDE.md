# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an unofficial fork of [InferenceX™](https://github.com/SemiAnalysisAI/InferenceX) that extends the official LLM inference benchmarking platform with **agentic coding workloads**. The fork adds three benchmark tests targeting multi-turn, KV-cache-sensitive scenarios:

1. **TTFT with caching** — ISL incremented 10k at a time (1k→51k), OSL fixed at 1k; tests cache efficiency across turns
2. **ITL memory bandwidth** — same ISL sweep; measures inter-token latency as context accumulates
3. **TTFT with delays** — same sweep with delays (0s, 1s, 5s, 10s, 1m) before each increment; tests cache eviction under bursty agentic patterns

## Commands

### Running Tests

```bash
cd utils
python -m pytest matrix_logic/ -v

# Single test file
python -m pytest matrix_logic/test_validation.py -v
python -m pytest matrix_logic/test_generate_sweep_configs.py -v

# Skip slow tests
python -m pytest matrix_logic/ -v -m "not slow"
```

### Generating and Validating Benchmark Configs

```bash
# Full sweep (also validates configs)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml

# Filter options (combinable): --model dsr1|gptoss, --framework sglang|trt|vllm|atom|dynamo-trt|dynamo-sglang|sglang-disagg, --precision fp4|fp8, --runner b200|h100|h200|gb200|mi300x|mi325x|mi355x
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml \
  --model dsr1 --framework sglang --precision fp8 --runner b200

# Generate with evals marked
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml --run-evals

# Only eval subset
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml --evals-only
```

### Processing Results

```bash
python utils/process_result.py
python utils/summarize.py
```

### Fetching GitHub Actions Benchmark Results

```bash
# List artifacts for a run
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'

# Download aggregated results
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results

# Parse results (avoid dumping raw JSON — it's large)
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)]
  | @tsv' | column -t
```

## Architecture

### Config-Driven Pipeline

The system is **configuration-driven**: YAML master configs define all benchmark parameters, Pydantic models validate them, and GitHub Actions orchestrate execution.

**End-to-end flow:**
1. Developer adds/updates entries in `.github/configs/nvidia-master.yaml` (or `amd-master.yaml`)
2. Developer adds a `perf-changelog.yaml` entry to trigger the benchmark
3. PR triggers `run-sweep.yml` → `process_changelog.py` parses which configs to run → `generate_sweep_configs.py` produces the matrix
4. Matrix jobs run via `benchmark-tmpl.yml` (single-node) or `benchmark-multinode-tmpl.yml` (multi-node)
5. Bash scripts in `benchmarks/` launch inference servers (vLLM/SGLang/TRT) and call `utils/bench_serving/benchmark_serving.py`
6. Optional evals via lm-eval; results aggregated by `collect-evals.yml`
7. `process_result.py` + `summarize.py` generate final artifacts

**Key files:**
- `.github/configs/nvidia-master.yaml` / `amd-master.yaml` — all benchmark definitions
- `utils/matrix_logic/validation.py` — Pydantic schemas (`SingleNodeMatrixEntry`, `MultiNodeMatrixEntry`)
- `utils/matrix_logic/generate_sweep_configs.py` — config expansion and eval marking logic
- `benchmarks/benchmark_lib.sh` — shared shell utilities (`wait_for_server_ready`, `run_benchmark_serving`, `run_eval`)
- `utils/bench_serving/benchmark_serving.py` — main benchmark client (41KB)
- `perf-changelog.yaml` — triggers which configs to benchmark

### Single-Node vs. Multi-Node (Disaggregated)

Multi-node configs (`multinode: true`, `disagg: true`) use separate prefill and decode workers. The Pydantic schemas split accordingly: single-node uses `SingleNodeSearchSpaceEntry`; multi-node uses `MultiNodeSearchSpaceEntry` with `prefill`/`decode` sub-configs.

For disaggregated configs (`dynamo-sglang`, `dynamo-trt`, `sglang-disagg`), recipes come from the external [srtslurm](https://github.com/ishandhanani/srt-slurm) repo, referenced via `additional-settings: CONFIG_FILE=recipes/...`.

### Evals

Evals are off by default (`RUN_EVAL=false`). When enabled, they run only on `1k8k` sequence length at two representative points per config group (lowest TP + highest conc, highest TP + highest conc). Eval task definitions live in `utils/evals/` (GSM8K, GPQA Diamond).

## Terminology

- **STP** — Single Token Prediction (standard autoregressive decoding, no speculation)
- **MTP** — Multi-Token Prediction (speculative decoding, e.g., EAGLE/NEXTN)
- **ISL/OSL** — Input/Output Sequence Length
- **TTFT** — Time to First Token
- **ITL** — Inter-Token Latency
- **Disagg** — Disaggregated inference: separate prefill and decode node pools

## Code Conventions

- **YAML field names**: kebab-case (`model-prefix`, `conc-start`, `dp-attn`)
- **Python**: Pydantic v2 with `extra='forbid'`, field aliases for YAML keys (`Field(alias="model-prefix")`)
- **Bash**: parameters via env vars; source `benchmark_lib.sh` for shared functions
- **Git**: use `[skip-sweep]` in commit message to skip benchmark runs

## Important Notes

- Do not create new directories in `/workspace` during benchmarks (files are OK)
- `perf-changelog.yaml` entries must be added whenever a config or Docker image is changed — this is what triggers benchmarks and tracks perf regressions
- The `experimental/` directory contains WIP Claude-generated code not used in official results
- This repo syncs upstream from `https://github.com/SemiAnalysisAI/InferenceX.git`
