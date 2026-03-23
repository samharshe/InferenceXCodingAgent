# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

## Commit 5 — chart_itl_bandwidth

Target: implement `chart_itl_bandwidth(results_dir)` in `benchmarks/visualize.py`.
The function currently just prints `"TODO: chart_itl_bandwidth"`. Replace it with a
full implementation, modelled exactly on `chart_ttft_caching` above it in the file.

### Steps

1. Open `benchmarks/visualize.py`. Find `chart_itl_bandwidth(results_dir)` at line 62.

2. Replace the stub body with the following logic (mirror `chart_ttft_caching` exactly,
   changing only the four items listed below):
   - File loaded per GPU: `{gpu}_itl-bandwidth.json` (was `{gpu}_ttft-caching.json`)
   - Metric extracted: `"itl_mean"` (was `"ttft_mean"`)
   - Chart title: `"ITL vs. Context Length (Memory Bandwidth)"` (was `"TTFT vs. Context Length (KV Cache Reuse)"`)
   - Y-axis label: `"Mean ITL (ms)"` (was `"Mean TTFT (ms)"`)
   - Warning message: `"no data for chart_itl_bandwidth, skipping"` (was `chart_ttft_caching`)
   - Output filename: `chart_itl_bandwidth.png` (was `chart_ttft_caching.png`)

   Everything else is identical: same GPUS list, same GPU_LABELS, same ISL_POSITIONS/
   ISL_LABELS x-ticks, same `figsize=(10, 6)`, same `dpi=150`, same `ax.legend()`,
   same `Written:` print, same skip-on-missing logic.

3. Smoke test — run the following commands:
   ```bash
   # Create a mock results dir with one itl-bandwidth file
   mkdir -p /tmp/mock_results

   python -c "
   import json
   turns = [{'turn': i, 'isl': 1000+i*10000, 'prefix_len': i*1000, 'new_tokens': 1000,
             'delay_s': 0, 'ttft_mean': 100.0+i*5, 'ttft_p50': 98.0, 'ttft_p99': 120.0,
             'itl_mean': 10.0+i*0.5, 'itl_p50': 9.8, 'itl_p99': 12.0, 'num_prompts': 20}
            for i in range(6)]
   print(json.dumps(turns, indent=2))
   " > /tmp/mock_raw.json

   python benchmarks/wrap_result.py \
     --input /tmp/mock_raw.json \
     --output /tmp/mock_results/h100_itl-bandwidth.json \
     --gpu h100 --test-type itl-bandwidth --delay-s 0 \
     --model openai/gpt-oss-120b --num-prompts 20 \
     --timestamp 2026-03-22T00:00:00

   python benchmarks/visualize.py /tmp/mock_results/
   ```

4. Verify output:
   - `Written: /tmp/mock_results/chart_itl_bandwidth.png` is printed.
   - `"TODO: chart_itl_bandwidth"` is NOT printed.
   - Warnings are printed for h200 and b200 (their files are missing) — that is expected.
   - Open the PNG and confirm it shows one line (H100) with correct x-axis labels
     (1k through 51k) and y-axis labelled "Mean ITL (ms)".

5. Commit with message: `Commit 5 — chart_itl_bandwidth`
