# SPEC.md
# Purpose: Precise engineering specification for the agentic coding benchmark execution,
# results storage, and visualization feature. Written 2026-03-22.

---

## Overview

This spec covers the second phase of the agentic coding benchmark feature. The first phase
(complete) built `benchmarks/agentic_coding.sh` and its supporting Pydantic validation. This
phase defines how to **run** those benchmarks across four GPU types, **store** the results
as structured JSON, and **visualize** them as three static PNG charts.

The workflow is deliberately manual and lightweight — a POC, not an automated pipeline.
The developer runs `agentic_coding.sh` on each AWS instance by hand, copies JSON back to
a local results directory, then runs a local Python visualization script.

---

## GPU Targets

Three GPU types, each as a single-node 8-GPU configuration. B300 excluded — no AWS instance type available.

| GPU   | Node config | HBM per node |
|-------|-------------|--------------|
| H100  | 8×H100 SXM  | 640 GB       |
| H200  | 8×H200 SXM  | 1,128 GB     |
| B200  | 8×B200 SXM  | 1,536 GB     |

GPU type is the **independent variable**. The same model, test types, and parameters are
used on every GPU. Results across GPUs are directly comparable.

---

## Model

**`openai/gpt-oss-120b`** (HuggingFace: `openai/gpt-oss-120b`)

- Architecture: Mixture of Experts, 117B total parameters, 5.1B active per forward pass
- Quantization: MXFP4 (fits on a single 80 GB GPU; well within any 8-GPU node)
- License: Apache 2.0
- Framework: vLLM
- Precision config in YAML: FP8 (existing convention in this repo)

This model is fixed across all four GPU types. It is chosen because it is state-of-the-art
for coding/reasoning tasks and is realistic for agentic coding workloads.

---

## Test Types

All three test types defined in `benchmarks/agentic_coding.sh` are run on every GPU:

1. **`ttft-caching`** — measures TTFT across six turns; expects flat TTFT if KV cache reuse works.
2. **`itl-bandwidth`** — measures ITL across six turns; reveals memory bandwidth pressure.
3. **`ttft-delays`** — measures TTFT with injected inter-turn delays simulating cache eviction.

For `ttft-delays`, five delay values are run as separate benchmark invocations:
`DELAY_S` ∈ {0, 1, 5, 10, 60}

This means **7 invocations per GPU** (2 standard test types + 5 delay values), **28 total
invocations** across all four GPUs.

Per-invocation parameters (all match existing agentic_coding.sh defaults):
- `NUM_PROMPTS=20`
- Six turns, ISL sweep: 1k, 11k, 21k, 31k, 41k, 51k tokens
- Output fixed at 1k tokens per turn

---

## Results Directory Structure

All results live under `benchmarks/results/` in the repository.

Each time the full suite is run, a new timestamped subdirectory is created:

```
benchmarks/results/
└── YYYY-MM-DDTHH:MM:SS/
    ├── h100_ttft-caching.json
    ├── h100_itl-bandwidth.json
    ├── h100_ttft-delays_0s.json
    ├── h100_ttft-delays_1s.json
    ├── h100_ttft-delays_5s.json
    ├── h100_ttft-delays_10s.json
    ├── h100_ttft-delays_60s.json
    ├── h200_ttft-caching.json
    ├── ... (same pattern for h200, b200, b300)
    └── b300_ttft-delays_60s.json
```

The timestamp is set once when the developer begins the run and used as the directory name
for all results from that suite execution.

---

## JSON Schema

Each result file is a JSON array of six turn objects (one per ISL step), exactly matching
the output format already produced by `agentic_coding.sh`. The only addition is a `meta`
top-level key wrapping the array, to carry run metadata.

### Full schema:

```json
{
  "meta": {
    "gpu": "h100",
    "test_type": "ttft-caching",
    "delay_s": 0,
    "model": "openai/gpt-oss-120b",
    "num_prompts": 20,
    "timestamp": "2026-03-22T14:30:00"
  },
  "turns": [
    {
      "turn": 0,
      "isl": 1000,
      "prefix_len": 0,
      "new_tokens": 1000,
      "delay_s": 0,
      "ttft_mean": 123.4,
      "ttft_p50": 120.1,
      "ttft_p99": 145.2,
      "itl_mean": 10.1,
      "itl_p50": 9.8,
      "itl_p99": 12.3,
      "num_prompts": 20
    },
    ...
  ]
}
```

`delay_s` in `meta` is 0 for `ttft-caching` and `itl-bandwidth` (field still present for
schema consistency). `gpu` is one of: `h100`, `h200`, `b200`, `b300`.

The developer is responsible for wrapping `agentic_coding.sh`'s raw output array into this
schema (adding the `meta` key) before saving. A thin wrapper script
(`benchmarks/wrap_result.py`) handles this — see below.

---

## wrap_result.py

A small Python script at `benchmarks/wrap_result.py` that takes the raw JSON array output
by `agentic_coding.sh` and wraps it with the `meta` block.

### CLI:

```bash
python benchmarks/wrap_result.py \
    --input /tmp/agentic_results.json \
    --output benchmarks/results/2026-03-22T14:30:00/h100_ttft-caching.json \
    --gpu h100 \
    --test-type ttft-caching \
    --delay-s 0 \
    --model openai/gpt-oss-120b \
    --num-prompts 20 \
    --timestamp 2026-03-22T14:30:00
```

All arguments are required (no defaults) so the developer is forced to be explicit.
The script reads the input JSON array, validates that it has exactly 6 elements, adds the
`meta` block, and writes the output file. Exits non-zero with a clear error message if
validation fails.

---

## visualize.py

A Python script at `benchmarks/visualize.py` that reads a results directory and produces
three PNG charts.

### CLI:

```bash
python benchmarks/visualize.py benchmarks/results/2026-03-22T14:30:00/
```

Single positional argument: path to the results subdirectory.

### Output:

Three PNG files written into the same results directory:

- `chart_ttft_caching.png`
- `chart_itl_bandwidth.png`
- `chart_ttft_delays.png`

### Chart specifications:

**Chart 1 — ttft_caching:**
- X-axis: ISL (tokens), values: 1000, 11000, 21000, 31000, 41000, 51000
- Y-axis: mean TTFT (ms)
- One line per GPU (H100, H200, B200, B300), distinguished by color
- Title: "TTFT vs. Context Length (KV Cache Reuse)"
- Legend showing GPU labels

**Chart 2 — itl_bandwidth:**
- X-axis: ISL (tokens)
- Y-axis: mean ITL (ms)
- One line per GPU, distinguished by color
- Title: "ITL vs. Context Length (Memory Bandwidth)"
- Legend showing GPU labels

**Chart 3 — ttft_delays:**
- X-axis: ISL (tokens)
- Y-axis: mean TTFT (ms)
- Lines vary by both GPU (line style: solid/dashed/dotted/dashdot) and delay value (color)
- Five colors for delay values: 0s, 1s, 5s, 10s, 60s
- Four line styles for GPUs: H100=solid, H200=dashed, B200=dotted, B300=dashdot
- Title: "TTFT vs. Context Length (Cache Eviction Under Delays)"
- Legend showing all GPU × delay combinations

### Implementation constraints:
- Pure matplotlib, no seaborn or other plotting libraries
- No interactivity (static PNGs only)
- If a GPU's result file is missing, skip that GPU with a printed warning — do not crash
- Figure size: 10×6 inches, 150 DPI
- All charts use the same x-axis tick labels: ["1k", "11k", "21k", "31k", "41k", "51k"]

---

## What Is Not In Scope

- Automated SSH orchestration across AWS instances (manual runs only)
- A Makefile or shell wrapper for visualization
- Interactive dashboards
- JSON validation beyond the 6-turn count check in wrap_result.py
- Any changes to `agentic_coding.sh`, `benchmark_lib.sh`, or the Pydantic validation code
- perf-changelog.yaml entries (those require real Docker image/model values, still placeholder)
