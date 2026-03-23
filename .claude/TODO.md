# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

## ONE-OFF FIX FIRST (Commit 4.5) — before starting Commit 5

**Target:** Fix `generate_sweep_configs.py` crash on agentic entries.

**The bug:** `generate_sweep_configs.py full-sweep` crashes with `KeyError: 'seq-len-configs'`
when it encounters any entry with `agentic: true`, because the loop unconditionally accesses
`val[Fields.SEQ_LEN_CONFIGS.value]` at line 160 with no guard for agentic entries.

**The fix:** In `generate_full_sweep` (around line 155–160 of
`utils/matrix_logic/generate_sweep_configs.py`), add an early `continue` for agentic entries
immediately after the existing `is_multinode` / `disagg` assignments and before the
`seq_len_configs = val[Fields.SEQ_LEN_CONFIGS.value]` line:

```python
if val.get('agentic', False):
    continue
```

**Constraints:**
- Touch ONLY `generate_sweep_configs.py`. No other file.
- The change must be exactly one guard — do not refactor anything else.
- After the fix, `python utils/matrix_logic/generate_sweep_configs.py full-sweep --config-files .github/configs/nvidia-master.yaml` must complete without error.
- `cd utils && python -m pytest matrix_logic/ -v` must still show 142 passing.
- Commit with `gacp "Fix generate_sweep_configs crash on agentic entries [skip-sweep]"`, then report back. Do NOT proceed to Commit 5.

---

Current target: Commit 5 — Scaffold `benchmarks/agentic_coding.sh` with env var validation and constants

## What this commit does

Create `benchmarks/agentic_coding.sh` — a new shell script that validates required environment
variables, declares hardcoded ISL sweep constants, and sets defaults. No turn loop yet. The script
exits cleanly after the validation and setup phase.

No changes to Python files. No changes to YAML files. No changes to existing shell scripts.
No new directories. One new file only: `benchmarks/agentic_coding.sh`.

## Exact steps

1. Look at `benchmarks/benchmark_lib.sh` and at least one existing script in `benchmarks/` to
   understand the conventions (env vars, sourcing pattern, shebang, etc.) before writing anything.

2. Create `benchmarks/agentic_coding.sh` with the following content (in order):
   - Shebang: `#!/usr/bin/env bash`
   - `set -euo pipefail`
   - Source `benchmark_lib.sh` from the same directory: `source "$(dirname "$0")/benchmark_lib.sh"`
   - Declare the three hardcoded ISL sweep arrays:
     ```bash
     ISL_VALUES=(1000 11000 21000 31000 41000 51000)
     PREFIX_LENS=(0 1000 11000 21000 31000 41000)
     NEW_TOKENS=(1000 10000 10000 10000 10000 10000)
     ```
   - Call `check_env_vars` to validate required inputs: `MODEL`, `API_URL`, `TEST_TYPE`, `RESULT_FILE`.
     Use whatever calling convention `benchmark_lib.sh` defines for `check_env_vars` — read the
     source before assuming.
   - Set defaults:
     ```bash
     NUM_PROMPTS=${NUM_PROMPTS:-20}
     DELAY_S=${DELAY_S:-0}
     ```
   - Validate `TEST_TYPE` is one of `ttft-caching`, `itl-bandwidth`, `ttft-delays`. If not, print
     an error message to stderr and `exit 1`.
   - Echo a startup message so it's clear the script reached the end of validation cleanly.

3. Mark the script executable: `chmod +x benchmarks/agentic_coding.sh`

4. Smoke-test: run the script with missing env vars and confirm it exits non-zero (don't actually
   run a benchmark). Example:
   ```bash
   bash benchmarks/agentic_coding.sh 2>&1 || echo "exited non-zero as expected"
   ```
   Also test with an invalid TEST_TYPE:
   ```bash
   MODEL=m API_URL=http://localhost:8000 TEST_TYPE=bad RESULT_FILE=/tmp/out.json \
     bash benchmarks/agentic_coding.sh 2>&1 || echo "exited non-zero as expected"
   ```

5. Run the test suite to confirm nothing broke: `cd utils && python -m pytest matrix_logic/ -v`
   All 142 tests must still pass.

6. Done. Report back. Do not proceed to Commit 6.

## Hard constraints

- Do NOT modify any existing file — not benchmark_lib.sh, not any Python file, not any YAML file,
  not any test file.
- Do NOT create any new directories.
- The ISL arrays must be exactly as specified above — these are hardcoded constants, not
  configurable via YAML or env vars.
- Source `benchmark_lib.sh` using a path relative to the script's own location (`$(dirname "$0")`),
  not a hardcoded absolute path.
- Use env vars for all inputs (MODEL, API_URL, etc.) — no CLI argument parsing.
- Read `benchmark_lib.sh` before writing anything. Do not assume what `check_env_vars` looks like.
