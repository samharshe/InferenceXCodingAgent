# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---


Current target: Commit 6 — Add the turn loop with per-turn `benchmark_serving.py` invocations

## What this commit does

Extend `benchmarks/agentic_coding.sh` (the only file you touch) with a six-turn loop that
calls `benchmark_serving.py` once per turn. After the loop, print a completion message.
No aggregation yet — that is Commit 7.

**No other files are modified.** Do not touch Python files, YAML files, `benchmark_lib.sh`,
or any test file.

## Exact steps

1. Read `utils/bench_serving/benchmark_serving.py` — specifically its CLI flags — before
   writing anything. You must confirm the exact flag names for:
   - specifying host/port or base URL
   - `--random-input-len`, `--random-prefix-len`, `--random-output-len`, `--random-range-ratio`
   - `--num-prompts`
   - saving results (`--save-result`, `--result-filename`, `--result-dir`)
   Do not guess flag names. Read the source.

2. Add the following block to `benchmarks/agentic_coding.sh` after the startup echo, replacing
   nothing that is already there:

   ```bash
   for t in 0 1 2 3 4 5; do
       if [[ $t -gt 0 && "${DELAY_S}" -gt 0 ]]; then
           sleep "$DELAY_S"
       fi

       TURN_RESULT_FILE="/tmp/agentic_turn_${t}.json"

       # Parse host and port from API_URL (e.g. http://hostname:8000)
       API_HOST=$(echo "$API_URL" | sed 's|https\?://||' | cut -d: -f1)
       API_PORT=$(echo "$API_URL" | sed 's|.*:||')

       python3 <path-to-benchmark_serving.py> \
           --backend openai \
           --host "$API_HOST" \
           --port "$API_PORT" \
           --model "$MODEL" \
           --random-input-len "${NEW_TOKENS[$t]}" \
           --random-prefix-len "${PREFIX_LENS[$t]}" \
           --random-output-len 1000 \
           --random-range-ratio 1.0 \
           --num-prompts "$NUM_PROMPTS" \
           --save-result \
           --result-filename "$TURN_RESULT_FILE"

       echo "Turn $t complete: ISL=${ISL_VALUES[$t]}"
   done

   echo "All turns complete."
   ```

   Use a path to `benchmark_serving.py` relative to the script's own location
   (`$(dirname "$0")/../utils/bench_serving/benchmark_serving.py`) — do NOT hardcode an
   absolute path.

   Verify the flag names against the actual CLI before committing. If the script uses
   `--result-filename` differently (e.g., requires no `.json` extension, or uses `--result-dir`
   separately), match the actual interface exactly.

3. Smoke-test — run the script with missing env vars to confirm validation still exits non-zero
   (the loop must not have broken the guard):
   ```bash
   bash benchmarks/agentic_coding.sh 2>&1 || echo "exited non-zero as expected"
   ```

4. Run the test suite to confirm nothing broke: `cd utils && python -m pytest matrix_logic/ -v`
   All 142 tests must still pass.

5. Done. Report back. Do not proceed to Commit 7.

## Hard constraints

- Modify ONLY `benchmarks/agentic_coding.sh`. Zero other files.
- Do NOT change the ISL arrays, env var validation, or startup echo already in the script.
- Parse `API_URL` into host and port inside the script — do not add new required env vars.
- Use a relative path to `benchmark_serving.py` anchored at `$(dirname "$0")`.
- Read `benchmark_serving.py`'s CLI before writing the invocation — flag names must be exact.
- The `DELAY_S` sleep must only fire when `t > 0` (no sleep before the first turn).
- Do NOT call `run_benchmark_serving` from `benchmark_lib.sh` — call `benchmark_serving.py`
  directly with `python3`. The lib wrapper has a different interface and different assumptions.
