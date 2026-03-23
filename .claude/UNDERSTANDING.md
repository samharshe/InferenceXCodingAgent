# UNDERSTANDING.md
# Purpose: Summary of recent agentic-coding-benchmarking feature development for any engineer
# picking up this work. Written 2026-03-22 after reviewing the full codebase and commit history.

---

## Overview

This repository is an unofficial fork of [InferenceX™](https://github.com/SemiAnalysisAI/InferenceX), the LLM inference benchmarking platform built by SemiAnalysis. The fork's sole purpose is to add **agentic coding workload benchmarks**: three tests that simulate the traffic pattern produced by AI coding agents like Claude Code, where each "turn" of a multi-turn conversation extends the shared context, stressing KV-cache reuse and memory bandwidth. The feature is implemented on the branch `feature/coding-agent-benchmarking` and consists of roughly 200 lines of new Python, 100 lines of shell, and three placeholder YAML config entries.

## What Was Built

Three benchmark tests were designed to answer the question: *how does an inference server perform as an agentic coding session accumulates context across six turns?* Each turn increments the input sequence length (ISL) by 10k tokens — from 1k up to 51k — while keeping the output fixed at 1k tokens. The three test types differ only in what they measure and whether delays are injected between turns:

1. **ttft-caching** — measures Time to First Token (TTFT) across all six turns; the expectation is that TTFT stays flat if the server efficiently reuses its KV cache rather than recomputing it.
2. **itl-bandwidth** — measures Inter-Token Latency (ITL) as context accumulates; this reveals memory bandwidth pressure since longer KV caches require more DRAM reads during decoding.
3. **ttft-delays** — same sweep but with a configurable delay (in seconds) inserted before each turn after the first; this simulates cache eviction under bursty agentic patterns (e.g., waiting for a tool call to return).

## Implementation Approach

The implementation follows the existing config-driven architecture of InferenceX. New benchmarks are registered in a YAML master config, validated by Pydantic models, and executed by a bash script that calls the existing `benchmark_serving.py` client. The implementation was planned as four phases across nine atomic commits, each narrow enough that a `git reset --hard` at any point would not damage working functionality.

## Phase 1: Pydantic Schema Extension (Commits 1–3)

The first phase extended `utils/matrix_logic/validation.py` with two new Pydantic models. `AgenticSearchSpaceEntry` defines the per-GPU-configuration search space for agentic runs — only `tp`, `ep`, and `dp-attn` fields, since agentic tests always run with concurrency=1 (no sweep). `AgenticMasterConfigEntry` defines the top-level config entry with required fields `image`, `model`, `model-prefix`, `precision`, `framework`, `runner`, `multinode: false`, `agentic: true`, `test-type`, and `search-space`, plus optional `num-prompts` (default 20) and `delays`. A `@model_validator` enforces that `delays` is present if and only if `test-type == 'ttft-delays'`.

The existing `validate_master_config` function was updated to dispatch on `entry.get('agentic', False)` before checking the multinode flag, routing agentic entries to `AgenticMasterConfigEntry`. Seven new tests were added to `test_validation.py` covering all valid and invalid cases. All 142 tests in the suite pass.

## Phase 2: YAML Config Entries (Commits 4 and 4.5)

Three placeholder YAML entries were added to `.github/configs/nvidia-master.yaml` — one for each test type, all targeting a B200 runner with vLLM and FP8 precision. The `image` and `model` fields are set to `<placeholder>` pending real Docker image and model path values. A matching guard was added to `generate_sweep_configs.py` (around line 155–157): agentic entries are silently skipped in the standard sweep via `if val.get('agentic', False): continue`, preventing a crash on the missing `seq-len-configs` key that standard entries require.

## Phase 3: Shell Script (Commits 5–7)

The core runtime artifact is `benchmarks/agentic_coding.sh`, a 99-line bash script that orchestrates the full six-turn benchmark. The script sources `benchmark_lib.sh`, declares three hardcoded arrays (`ISL_VALUES`, `PREFIX_LENS`, `NEW_TOKENS`), validates four required environment variables (`MODEL`, `API_URL`, `TEST_TYPE`, `RESULT_FILE`), and enforces that `TEST_TYPE` is one of the three allowed values. One notable workaround: `check_env_vars` in `benchmark_lib.sh` uses indirect variable expansion (`${!var_name}`) which is incompatible with `set -u`; the script temporarily disables nounset before calling it and re-enables it afterward. This is documented in `LOOSE_ENDS.md`.

The turn loop runs for turns 0 through 5. For each turn, it sleeps `$DELAY_S` seconds (if > 0 and t > 0), then calls `benchmark_serving.py` with `--dataset-name random`, the per-turn input length (`NEW_TOKENS[$t]`), prefix length (`PREFIX_LENS[$t]`), and output length (fixed at 1000), writing results to `/tmp/agentic_turn_${t}.json`. After the loop, an inline Python block reads all six JSON files and writes a single aggregated output list to `$RESULT_FILE`, mapping benchmark_serving.py's output keys (`mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`, `mean_itl_ms`, `median_itl_ms`, `p99_itl_ms`) to the spec's field names (`ttft_mean`, `ttft_p50`, `ttft_p99`, `itl_mean`, `itl_p50`, `itl_p99`). Temp files are cleaned up afterward.

## Phase 4: Verification (Commit 8)

Commit 8 added a comment block above the aggregation code documenting the verified output key names from `benchmark_serving.py`. The keys were confirmed by tracing through the source: the `process_one_metric` function at line 551 emits `median_{metric}_ms` for the p50 percentile (not `p50_{metric}_ms`), and `p99_{metric}_ms` for p99. No mock server was available, so the smoke test consisted of: (1) confirming env-var-missing exits non-zero, (2) inspection of all CLI flags against the `benchmark_serving.py` argument parser, and (3) running the full 142-test suite. All passed.

## Constraint Satisfaction

All constraints specified in SPEC.md and enforced in PLAN.md appear satisfied:

- **No new directories** — only new files (`benchmarks/agentic_coding.sh`, three `.claude/` docs).
- **No existing classes modified** — `validate_master_config` was updated (dispatch added), but no existing Pydantic model class was changed.
- **bash parameters via env vars** — the script reads only env vars, consistent with `benchmark_lib.sh` convention.
- **JSON field names match spec** — the aggregation block maps to `ttft_mean/p50/p99` and `itl_mean/p50/p99` exactly as specified.
- **142 tests pass** — confirmed as of Commit 8.
- **`[skip-sweep]` tags** — all implementation commits include this tag to prevent triggering real benchmark runs.
- **extra='forbid'** — both new Pydantic models use `ConfigDict(extra='forbid')`.
- **delays cross-validation** — the `@model_validator` correctly enforces the delays ↔ test_type relationship.

## What Is Not Yet Complete

The YAML entries in `nvidia-master.yaml` use `image: <placeholder>` and `model: <placeholder>`. These must be replaced with a real Docker image name and a real model path before the benchmark can actually run. This is by design — the SPEC notes that real values were not available during development. Additionally, `perf-changelog.yaml` has not been updated with agentic entries, which means GitHub Actions will not trigger these benchmarks on a PR even once the placeholders are replaced. Adding `perf-changelog.yaml` entries is the natural next step after filling in the real values.

## Architecture: Client vs. Server

`agentic_coding.sh` is purely a **client script**. It never starts an inference server — it calls an already-running one via HTTP through `benchmark_serving.py`. The two are completely decoupled:

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│  agentic_coding.sh          │  HTTP   │  Inference server        │
│  (runs on host)             │────────▶│  (vLLM / SGLang / etc.)  │
│  calls benchmark_serving.py │         │  loaded with your model  │
└─────────────────────────────┘         └──────────────────────────┘
```

The `image` field in the YAML configs is **only used by GitHub Actions CI** — the workflow pulls the Docker image and launches the inference server inside it on a bare-metal GPU runner. For local or manual runs, the `image` field is irrelevant.

## How to Use the Current Repo

### Running locally or on AWS

You do **not** need to build Docker images. Start any OpenAI-compatible inference server (vLLM, SGLang, etc.), then point the script at it:

```bash
# 1. Start the inference server (needs a GPU; can be local or a remote AWS instance)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# 2. Install benchmark_serving.py dependencies
pip install openai aiohttp

# 3. Run the benchmark
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export API_URL="http://localhost:8000"   # or http://<aws-instance-ip>:8000
export TEST_TYPE="ttft-caching"          # or itl-bandwidth, ttft-delays
export RESULT_FILE="/tmp/agentic_results.json"
export NUM_PROMPTS=20                    # optional, default 20
export DELAY_S=0                         # optional; only meaningful for ttft-delays
bash benchmarks/agentic_coding.sh
```

The script runs six turns, prints progress, writes aggregated results to `$RESULT_FILE` as a JSON array, and cleans up temp files. For `ttft-delays`, set `DELAY_S` to the desired inter-turn delay in seconds (0, 1, 5, 10, or 60 — one run per delay value).

### Enabling CI (GitHub Actions)

The three YAML entries in `nvidia-master.yaml` currently have `image: <placeholder>` and `model: <placeholder>`. Before CI can trigger these runs automatically:

1. Replace the placeholders with a real Docker image (e.g., an NVIDIA vLLM image) and a real model path.
2. Add a corresponding `perf-changelog.yaml` entry — this is what triggers GitHub Actions to schedule the run.

Until then, CI will not run the agentic benchmarks, but manual execution works as described above.

### Validating configs and running tests

```bash
# Validate YAML (agentic entries are skipped in sweep output, but validated):
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml

# Run full test suite (should report 142 passed):
cd utils && python -m pytest matrix_logic/ -v
```

## Commit History Summary

| Commit | Description |
|--------|-------------|
| `91a08c3` | Planning docs: mark Commit 8 done, write Commit 9 spec |
| `888aad9` | Commit 8: add verified-key comment to agentic_coding.sh |
| `c19fdc5` | Commit 7: result aggregation into single JSON |
| `0f31d94` | Commit 6: six-turn loop with benchmark_serving.py invocations |
| `c090461` | Commit 5: scaffold agentic_coding.sh with env validation |
| `dc24c67` | Commit 4.5: fix generate_sweep_configs crash on agentic entries |
| `f40c147` | Commit 4: add agentic YAML config entries |
| `983f9b0` | Commit 3: validation tests |
| (earlier) | Commits 1–2: Pydantic schemas and dispatch |
