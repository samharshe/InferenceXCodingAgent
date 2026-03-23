# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---


Current target: Commit 10 — Fix `check_env_vars` to be compatible with `set -u`

## What this commit does

A single surgical change to `benchmarks/benchmark_lib.sh`. Replace the indirect
expansion `${!var_name}` in `check_env_vars` with `${!var_name+x}`, which is
nounset-safe and returns `x` if the variable is set (even if empty) or empty string
if unset — without triggering `set -u`. Once this is fixed, the `set +u` / `set -u`
workaround in `benchmarks/agentic_coding.sh` can be removed.

Two files change, nothing else.

## Exact steps

1. **Edit `benchmarks/benchmark_lib.sh`**: In `check_env_vars`, change the guard
   from:
   ```bash
   if [[ -z "${!var_name}" ]]; then
   ```
   to:
   ```bash
   if [[ -z "${!var_name+x}" ]]; then
   ```
   This makes the check nounset-safe: `${!var_name+x}` expands to `x` when the
   variable is set (to any value, including empty) and to `` when it is unset —
   so `-z` correctly catches only the unset case without triggering `set -u`.

2. **Edit `benchmarks/agentic_coding.sh`**: Remove the `set +u` / `set -u`
   workaround lines (currently lines 12–16):
   ```bash
   # check_env_vars uses ${!var_name} (indirect expansion) which is incompatible with set -u;
   # temporarily disable nounset so the function can detect and report missing vars itself.
   set +u
   check_env_vars MODEL API_URL TEST_TYPE RESULT_FILE
   set -u
   ```
   Replace with the single line:
   ```bash
   check_env_vars MODEL API_URL TEST_TYPE RESULT_FILE
   ```
   Also remove the comment above it — it is no longer true.

3. **Update `LOOSE_ENDS.md`**: Mark the `check_env_vars` / `set -u` entry as
   resolved. Do not delete it — add a one-line note at the bottom of that entry:
   `**Resolved in Commit 10.**`

4. **Smoke-test**: Run:
   ```bash
   # Missing env var should still exit non-zero
   bash benchmarks/agentic_coding.sh
   ```
   Expect: "Error: The following required environment variables are not set: MODEL
   API_URL TEST_TYPE RESULT_FILE" and exit code 1.

5. **Run test suite**: `cd utils && python -m pytest matrix_logic/ -v`
   All 142 tests must pass. (These tests do not exercise the shell script, but
   run them anyway to confirm nothing else was disturbed.)

6. Done. Report back.

## Hard constraints

- Touch ONLY `benchmarks/benchmark_lib.sh`, `benchmarks/agentic_coding.sh`,
  and `.claude/LOOSE_ENDS.md`.
- Do NOT change any logic in `check_env_vars` beyond the one-character fix
  (`-z "${!var_name}"` → `-z "${!var_name+x}"`).
- Do NOT refactor, reformat, or rename anything else in either shell file.
- Do NOT add new tests — the smoke test above is sufficient.
