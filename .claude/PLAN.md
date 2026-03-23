# PLAN.md
# Purpose: Step-by-step plan to achieve the development goal. Each item corresponds to a single Git commit.
# Derived from SPEC.md once the spec is finalized.

---

## Phase 1: Pydantic Schema Extension

Everything downstream (YAML validation, tests) depends on the schema being correct first. This phase establishes the data contracts. It is pure Python and fully testable without a running server.

### Morning 1

**~~Commit 1 — Add `AgenticSearchSpaceEntry` and `AgenticMasterConfigEntry` to `validation.py`~~ ✓ DONE**

- Open `utils/matrix_logic/validation.py`.
- Add `AgenticSearchSpaceEntry` after the existing search-space classes:
  ```python
  class AgenticSearchSpaceEntry(BaseModel):
      model_config = ConfigDict(extra='forbid', populate_by_name=True)
      tp: int
      ep: Optional[int] = None
      dp_attn: Optional[bool] = Field(default=None, alias='dp-attn')
  ```
- Add `AgenticMasterConfigEntry` after the existing master-config classes:
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
      delays: Optional[List[int]] = None
      search_space: List[AgenticSearchSpaceEntry] = Field(alias='search-space')

      @model_validator(mode='after')
      def validate_delays(self):
          if self.test_type == 'ttft-delays' and not self.delays:
              raise ValueError("'delays' is required when test-type is 'ttft-delays'")
          if self.test_type != 'ttft-delays' and self.delays:
              raise ValueError("'delays' is only valid when test-type is 'ttft-delays'")
          return self
  ```
- Do NOT modify any existing classes.

**~~Commit 2 — Update `validate_master_config` to dispatch agentic entries~~ ✓ DONE**

- In `validate_master_config`, add agentic dispatch before the existing multinode check:
  ```python
  is_agentic = entry.get('agentic', False)
  if is_agentic:
      AgenticMasterConfigEntry(**entry)
  elif is_multinode:
      MultiNodeMasterConfigEntry(**entry)
  else:
      SingleNodeMasterConfigEntry(**entry)
  ```
- This must not touch the existing dispatch logic for non-agentic entries.
- Run existing tests: `cd utils && python -m pytest matrix_logic/ -v` — all must pass.

**~~Commit 3 — Write Pydantic validation tests~~ ✓ DONE**

- Add tests to `utils/matrix_logic/test_validation.py` (import the new classes at the top).
- Test cases to cover:
  1. Valid `ttft-caching` entry — passes without error.
  2. Valid `itl-bandwidth` entry — passes without error.
  3. Valid `ttft-delays` entry with `delays: [0, 1, 5, 10, 60]` — passes without error.
  4. `ttft-delays` with no `delays` field — raises `ValueError`.
  5. `ttft-caching` with a `delays` field present — raises `ValueError`.
  6. `AgenticMasterConfigEntry` with unknown extra field — raises `ValidationError` (extra='forbid').
  7. `validate_master_config` called with a mixed dict (one agentic entry, one standard single-node entry) — both validate without error.
- Run: `cd utils && python -m pytest matrix_logic/test_validation.py -v` — all new tests green.

---

## Phase 2: YAML Config Entries

Add real config entries to the master YAML. Validate they pass end-to-end via the existing `generate_sweep_configs.py` tool (which calls `validate_master_config` internally). This is the integration check for Phase 1.

### Afternoon 1

**~~Commit 4 — Add agentic config entries to `nvidia-master.yaml`~~ ✓ DONE**

- Open `.github/configs/nvidia-master.yaml`.
- Append three placeholder entries (use a placeholder image and model path — mark with `# TODO: replace with real values`):
  ```yaml
  gptoss-fp8-b200-vllm-agentic-ttft-caching:
    image: <placeholder>
    model: <placeholder>
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

  gptoss-fp8-b200-vllm-agentic-itl-bandwidth:
    image: <placeholder>
    model: <placeholder>
    model-prefix: gptoss
    precision: fp8
    framework: vllm
    runner: b200
    multinode: false
    agentic: true
    test-type: itl-bandwidth
    num-prompts: 20
    search-space:
      - tp: 1

  gptoss-fp8-b200-vllm-agentic-ttft-delays:
    image: <placeholder>
    model: <placeholder>
    model-prefix: gptoss
    precision: fp8
    framework: vllm
    runner: b200
    multinode: false
    agentic: true
    test-type: ttft-delays
    delays: [0, 1, 5, 10, 60]
    num-prompts: 20
    search-space:
      - tp: 1
  ```
- Validate: `python utils/matrix_logic/generate_sweep_configs.py full-sweep --master-config .github/configs/nvidia-master.yaml` — must not raise errors on the new entries. (The existing generate logic will skip them since they have no `seq-len-configs` — that's expected and acceptable at this stage; we only care that `validate_master_config` does not blow up.)
- Run full test suite: `cd utils && python -m pytest matrix_logic/ -v` — all green.

---

## Phase 3: Shell Script

The shell script is the core runtime artifact. It has no unit tests — correctness is verified by dry-running with a mock server endpoint or by inspection. Write it incrementally.

### Morning 2

**Commit 5 — Scaffold `benchmarks/agentic_coding.sh` with env var validation and constants**

- Create `benchmarks/agentic_coding.sh`.
- Source `benchmark_lib.sh` at the top (same directory).
- Declare the hardcoded ISL sweep constants as bash arrays:
  ```bash
  ISL_VALUES=(1000 11000 21000 31000 41000 51000)
  PREFIX_LENS=(0 1000 11000 21000 31000 41000)
  NEW_TOKENS=(1000 10000 10000 10000 10000 10000)
  ```
- Call `check_env_vars MODEL API_URL TEST_TYPE RESULT_FILE` to validate required inputs.
- Set defaults: `NUM_PROMPTS=${NUM_PROMPTS:-20}`, `DELAY_S=${DELAY_S:-0}`.
- Validate `TEST_TYPE` is one of `ttft-caching`, `itl-bandwidth`, `ttft-delays`. Exit 1 otherwise.
- No turn loop yet. Script exits cleanly after validation. Mark file executable (`chmod +x`).

**Commit 6 — Add the turn loop with per-turn `benchmark_serving.py` invocations**

- Add a `for t in 0 1 2 3 4 5` loop.
- For each turn:
  1. If `DELAY_S > 0` and `t > 0`: `sleep "$DELAY_S"`.
  2. Set `TURN_RESULT_FILE` to a temp path (e.g., `/tmp/agentic_turn_${t}.json`).
  3. Call `benchmark_serving.py` with:
     - `--backend openai` (matches existing lib convention)
     - `--host` and `--port` parsed from `API_URL`
     - `--model "$MODEL"`
     - `--random-input-len "${NEW_TOKENS[$t]}"`
     - `--random-prefix-len "${PREFIX_LENS[$t]}"`
     - `--random-output-len 1000`
     - `--random-range-ratio 1.0`
     - `--num-prompts "$NUM_PROMPTS"`
     - `--save-result`
     - `--result-filename "$TURN_RESULT_FILE"`
  4. Echo progress: `echo "Turn $t complete: ISL=${ISL_VALUES[$t]}"`.
- After loop, echo "All turns complete."

**Commit 7 — Add result aggregation into a single JSON output file**

- After the turn loop, aggregate the six per-turn JSON files into `$RESULT_FILE`.
- Use Python inline (`python3 -c "..."`) to:
  1. Load each `/tmp/agentic_turn_${t}.json`.
  2. Extract fields: `mean_ttft_ms`, `p50_ttft_ms`, `p99_ttft_ms`, `mean_itl_ms`, `p50_itl_ms`, `p99_itl_ms` (exact keys from `benchmark_serving.py` output — verify against the script before writing).
  3. Build a per-turn object with keys: `turn`, `isl`, `prefix_len`, `new_tokens`, `delay_s`, `ttft_mean`, `ttft_p50`, `ttft_p99`, `itl_mean`, `itl_p50`, `itl_p99`, `num_prompts`.
  4. Write the array to `$RESULT_FILE`.
- Clean up temp files.
- Echo "Results written to $RESULT_FILE."

---

## Phase 4: Verification and Polish

These commits do not add features — they catch mistakes made in Phase 3.

### Afternoon 2

**Commit 8 — Smoke-test the shell script against a local mock server (dry run)**

- This commit documents the outcome of manual testing.
- If the script runs end-to-end (even with a mock server returning dummy JSON), confirm all six temp files are created and the aggregated JSON is valid.
- Fix any discovered bugs. The fix goes in the same commit as the test outcome notes (add a comment in the script if needed).
- If the mock server cannot be run in this environment, verify by inspecting the script logic against `benchmark_serving.py`'s CLI flags and JSON output structure — document the verified field names in a comment at the top of the aggregation block.

**Commit 9 — Update `TODO.md` and `PLAN.md` to reflect completion**

- Mark all steps done.
- Note any loose ends discovered during implementation in `LOOSE_ENDS.md`.

---

## Invariants (apply to every commit)

- `cd utils && python -m pytest matrix_logic/ -v` must pass after every commit.
- No existing class in `validation.py` is modified.
- No new directories are created (only files).
- Every bash script sources `benchmark_lib.sh` and uses env vars (not CLI args).
- JSON field names in the result file match exactly what the spec defines.
