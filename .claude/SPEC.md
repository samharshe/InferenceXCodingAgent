# SPEC.md
# Purpose: Precise description of the ultimate development goal — what must be built, how it must behave,
# and what success looks like. This is the source of truth for implementation decisions.

---

## 1. Overview

This project adds three benchmark tests to the InferenceX platform that measure LLM inference performance under **multi-turn agentic coding workloads** — traffic patterns representative of tools like Claude Code, where each turn appends accumulated context (prior turns + tool outputs) to the prompt.

The tests are driven by the existing `utils/bench_serving/benchmark_serving.py` client, orchestrated by a new shell script, and defined via new entries in the YAML master config with a new Pydantic schema.

GitHub Actions integration is **out of scope**.

---

## 2. The Three Tests

All three tests share the same ISL sweep and OSL:

- **ISL sweep**: 1k → 51k tokens, incrementing 10k per turn → 6 turns at ISLs: 1k, 11k, 21k, 31k, 41k, 51k
- **OSL**: 1k tokens per turn (fixed)
- **Concurrency**: 1 (single session; no concurrent requests)

### Test 1 — TTFT with KV Cache Reuse (`ttft-caching`)

Measures TTFT at each turn. Each successive turn sends the full accumulated context (prefix from all prior turns + 10k new tokens). The prefix is identical across turns, so a caching-capable server should avoid recomputing it. The test measures whether TTFT stays flat or degrades as ISL grows.

**No delays between turns.**

### Test 2 — ITL Memory Bandwidth (`itl-bandwidth`)

Measures ITL (inter-token latency) at each turn with the same ISL sweep. As the KV cache grows, memory bandwidth pressure increases. The test reveals whether ITL degrades as the KV cache footprint grows across turns.

**No delays between turns.**

### Test 3 — TTFT with Cache Eviction (`ttft-delays`)

Same as Test 1, but introduces a configurable **delay before each turn** to simulate bursty agentic patterns where the server may evict KV cache entries between turns. The test is run separately for each delay duration:

- **Delays**: 0s, 1s, 5s, 10s, 60s

Each delay value produces its own benchmark run (i.e., 5 runs total for Test 3, one per delay).

---

## 3. Implementation: Shell Script Orchestration

A new script `benchmarks/agentic_coding.sh` orchestrates the multi-turn simulation. It does not modify `benchmark_serving.py`.

### Turn Loop Logic

For each turn `t` in {0, 1, 2, 3, 4, 5}:

```
isl_values  = [1000, 11000, 21000, 31000, 41000, 51000]
prefix_lens = [   0,  1000, 11000, 21000, 31000, 41000]
new_tokens  = [1000, 10000, 10000, 10000, 10000, 10000]
```

For turn `t`:
1. If `DELAY_S > 0` and `t > 0`: `sleep $DELAY_S`
2. Call `benchmark_serving.py` with:
   - `--random-input-len ${new_tokens[t]}`
   - `--random-prefix-len ${prefix_lens[t]}`
   - `--random-output-len 1000`
   - `--random-range-ratio 1.0` (no length variance)
   - `--num-prompts $NUM_PROMPTS` (default: 20)
   - `--save-result`
   - `--result-filename` set to a per-turn temp file
3. Capture per-turn metrics from the result file.

After all turns complete, the script aggregates per-turn result JSON objects into a single output file.

### Environment Variables (inputs to the script)

| Variable | Description |
|---|---|
| `MODEL` | Model path / name |
| `API_URL` | Inference server URL |
| `NUM_PROMPTS` | Requests per turn (default: 20) |
| `DELAY_S` | Seconds to sleep between turns (default: 0; Test 3 only) |
| `TEST_TYPE` | `ttft-caching`, `itl-bandwidth`, or `ttft-delays` |
| `RESULT_FILE` | Path to write the final aggregated JSON |

These follow the existing convention in `benchmarks/benchmark_lib.sh` (env vars, not CLI args).

---

## 4. Results Format

The script produces a **single aggregated JSON file** at `$RESULT_FILE`. It is an array of per-turn result objects:

```json
[
  {
    "turn": 0,
    "isl": 1000,
    "prefix_len": 0,
    "new_tokens": 1000,
    "delay_s": 0,
    "ttft_mean": ...,
    "ttft_p50": ...,
    "ttft_p99": ...,
    "itl_mean": ...,
    "itl_p50": ...,
    "itl_p99": ...,
    "num_prompts": 20
  },
  ...
]
```

All fields are present for all three test types. `delay_s` is 0 for Tests 1 and 2. TTFT metrics are the primary signal for Tests 1 and 3; ITL metrics are the primary signal for Test 2.

---

## 5. YAML Config Format

New entries in `.github/configs/nvidia-master.yaml` follow a new schema for agentic tests. They are **separate from existing `SingleNodeMasterConfigEntry` entries** — they do not use `seq-len-configs`.

Example entry:

```yaml
gptoss-fp8-b200-vllm-agentic-ttft-caching:
  image: <docker-image>
  model: <model-path>
  model-prefix: gptoss
  precision: fp8
  framework: vllm
  runner: b200
  multinode: false
  agentic: true
  test-type: ttft-caching
  num-prompts: 20
  search-space:
    - tp: 1
    - tp: 2
```

For `ttft-delays`, a `delays` list is added:

```yaml
gptoss-fp8-b200-vllm-agentic-ttft-delays:
  ...
  test-type: ttft-delays
  delays: [0, 1, 5, 10, 60]
  search-space:
    - tp: 1
```

The ISL sweep (1k→51k, step 10k) and OSL (1k) are **hardcoded constants** in the shell script and Pydantic models — they are not configurable via YAML. They define the benchmark; there is no reason to vary them.

---

## 6. Pydantic Schema Changes (`validation.py`)

New classes are added to `validation.py`. Existing classes are not modified.

### New: `AgenticSearchSpaceEntry`

```python
class AgenticSearchSpaceEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    tp: int
    ep: Optional[int] = None
    dp_attn: Optional[bool] = Field(default=None, alias='dp-attn')
```

No concurrency fields (concurrency is always 1 for agentic tests).

### New: `AgenticMasterConfigEntry`

```python
class AgenticMasterConfigEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    image: str
    model: str
    model_prefix: str = Field(alias='model-prefix')
    precision: str
    framework: str
    runner: str
    multinode: Literal[False]
    agentic: Literal[True]
    test_type: Literal['ttft-caching', 'itl-bandwidth', 'ttft-delays'] = Field(alias='test-type')
    num_prompts: int = Field(default=20, alias='num-prompts')
    delays: Optional[List[int]] = None  # required when test_type == 'ttft-delays'
    search_space: List[AgenticSearchSpaceEntry] = Field(alias='search-space')

    @model_validator(mode='after')
    def validate_delays(self):
        if self.test_type == 'ttft-delays' and not self.delays:
            raise ValueError("'delays' is required when test-type is 'ttft-delays'")
        if self.test_type != 'ttft-delays' and self.delays:
            raise ValueError("'delays' is only valid when test-type is 'ttft-delays'")
        return self
```

### Updated: `validate_master_config`

The existing `validate_master_config` function is updated to dispatch to `AgenticMasterConfigEntry` when `entry.get('agentic') == True`, falling back to existing logic otherwise.

---

## 7. Success Criteria

The implementation is complete when:

1. `benchmarks/agentic_coding.sh` runs end-to-end against a live inference server, producing a valid aggregated JSON result file for each of the three test types.
2. New YAML config entries in `nvidia-master.yaml` pass `validate_master_config` without errors.
3. Existing tests in `utils/matrix_logic/` continue to pass unchanged.
4. New Pydantic validation tests cover: valid agentic configs, missing `delays` on `ttft-delays`, unexpected `delays` on non-delays tests.
