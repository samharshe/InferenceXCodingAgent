# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---


Current target: Commit 8 — Smoke-test the shell script and document verified field names

## What this commit does

Verify `benchmarks/agentic_coding.sh` end-to-end by inspecting the script
logic against `benchmark_serving.py`'s CLI flags and JSON output structure.
Document the verified field names in a comment at the top of the aggregation
block inside the script. If any bugs are found, fix them in the same commit.

**The ONLY file you touch is `benchmarks/agentic_coding.sh`.** Do not touch
Python files, YAML files, `benchmark_lib.sh`, test files, or any other file.

## Exact steps

1. **Read the current state of `benchmarks/agentic_coding.sh`** in full before
   touching anything.

2. **Verify the following by cross-referencing `benchmark_serving.py`:**
   - Every CLI flag passed to `benchmark_serving.py` is valid (grep the arg
     parser to confirm `--backend`, `--host`, `--port`, `--model`,
     `--dataset-name`, `--random-input-len`, `--random-prefix-len`,
     `--random-output-len`, `--random-range-ratio`, `--num-prompts`,
     `--save-result`, `--result-filename` all exist).
   - The JSON output keys used in the aggregation block (`mean_ttft_ms`,
     `median_ttft_ms`, `p99_ttft_ms`, `mean_itl_ms`, `median_itl_ms`,
     `p99_itl_ms`) are actually written by `benchmark_serving.py` under
     default `--percentile-metrics` and `--metric-percentiles` settings.
   - The `$RESULT_FILE` is written via `open()` in Python, not stdout redirect.
   - The temp file cleanup loop covers exactly turns 0–5.

3. **Add a comment block** immediately above the `python3 -c "..."` aggregation
   line (after `echo "All turns complete."`). The comment must read exactly:

   ```bash
   # Verified output keys from benchmark_serving.py (default --percentile-metrics=ttft,tpot,itl
   # and --metric-percentiles=99): mean_ttft_ms, median_ttft_ms (p50), p99_ttft_ms,
   # mean_itl_ms, median_itl_ms (p50), p99_itl_ms.
   ```

   Do NOT modify any logic — only add this comment.

4. **Smoke-test** — confirm the guard still exits non-zero with missing env vars:
   ```bash
   bash benchmarks/agentic_coding.sh 2>&1 || echo "exited non-zero as expected"
   ```

5. **Run the test suite**: `cd utils && python -m pytest matrix_logic/ -v`
   All 142 tests must still pass.

6. Done. Report back. Do not proceed to Commit 9.

## Hard constraints

- Modify ONLY `benchmarks/agentic_coding.sh`. Zero other files.
- The only change to the script is adding the comment block in step 3.
  If verification in step 2 reveals a bug, fix it — but explain the bug
  clearly before patching it.
- Do NOT reformat, reorder, or touch any existing lines outside the comment.
- Do NOT add a separate `.py` file or any other artifact.
