# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

## Commit 6 — chart_ttft_delays (lines + single legend)

Target: implement `chart_ttft_delays(results_dir)` in `benchmarks/visualize.py`.
The function currently just prints `"TODO: chart_ttft_delays"`. Replace it with a
working implementation that plots all GPU×delay combinations with correct styling.

**CRITICAL CONSTRAINTS — do not deviate:**
- This commit ends with a SINGLE combined legend. The dual-legend upgrade is Commit 7.
  Do not implement two legends here.
- Do not modify `chart_ttft_caching`, `chart_itl_bandwidth`, or any other function.
- Do not add any new module-level constants or imports beyond what is needed.

---

### Steps

1. Open `benchmarks/visualize.py`. Find `chart_ttft_delays(results_dir)` near the
   bottom of the file (currently a one-line stub).

2. Define these two local variables at the top of the function body:

   ```python
   DELAYS = [0, 1, 5, 10, 60]
   GPU_LINESTYLES = {"h100": "solid", "h200": "dashed", "b200": "dotted"}
   ```

3. Create the figure:
   ```python
   fig, ax = plt.subplots(figsize=(10, 6))
   any_data = False
   ```

4. Loop over GPUs then delays. For each (gpu, delay) pair:
   - Load `{gpu}_ttft-delays_{delay}s.json` using `load_file`.
   - If `None`, print warning and `continue`.
   - Extract isl and ttft_mean using `get_isl_and_metric(data["turns"], "ttft_mean")`.
   - Determine color: `f"C{DELAYS.index(delay)}"` (gives C0–C4 for tab10 defaults).
   - Determine linestyle: `GPU_LINESTYLES[gpu]`.
   - Plot: `ax.plot(isl, ttft, color=color, linestyle=linestyle, label=f"{GPU_LABELS[gpu]} {delay}s")`.
   - Set `any_data = True`.

5. If `not any_data`: print warning, `plt.close(fig)`, return.

6. Set axes (identical to other charts):
   ```python
   ax.set_xticks(ISL_POSITIONS)
   ax.set_xticklabels(ISL_LABELS)
   ax.set_title("TTFT vs. Context Length (Cache Eviction Under Delays)")
   ax.set_xlabel("Input Sequence Length (tokens)")
   ax.set_ylabel("Mean TTFT (ms)")
   ax.legend()
   ```

7. Save and print:
   ```python
   out = os.path.join(results_dir, "chart_ttft_delays.png")
   fig.savefig(out, dpi=150)
   plt.close(fig)
   print(f"Written: {out}")
   ```

---

### Smoke test

```bash
mkdir -p /tmp/mock_delays

# Generate mock raw JSON once
python -c "
import json
turns = [{'turn': i, 'isl': 1000+i*10000, 'prefix_len': i*1000, 'new_tokens': 1000,
          'delay_s': 0, 'ttft_mean': 100.0+i*5, 'ttft_p50': 98.0, 'ttft_p99': 120.0,
          'itl_mean': 10.0+i*0.5, 'itl_p50': 9.8, 'itl_p99': 12.0, 'num_prompts': 20}
         for i in range(6)]
print(json.dumps(turns, indent=2))
" > /tmp/mock_raw.json

# Wrap two files: h100 delay=0 and h100 delay=60
python benchmarks/wrap_result.py \
  --input /tmp/mock_raw.json \
  --output /tmp/mock_delays/h100_ttft-delays_0s.json \
  --gpu h100 --test-type ttft-delays --delay-s 0 \
  --model openai/gpt-oss-120b --num-prompts 20 \
  --timestamp 2026-03-22T00:00:00

python benchmarks/wrap_result.py \
  --input /tmp/mock_raw.json \
  --output /tmp/mock_delays/h100_ttft-delays_60s.json \
  --gpu h100 --test-type ttft-delays --delay-s 60 \
  --model openai/gpt-oss-120b --num-prompts 20 \
  --timestamp 2026-03-22T00:00:00

python benchmarks/visualize.py /tmp/mock_delays/
```

**Expected output:**
- Warnings for all missing files (h200, b200, delays 1/5/10 for h100) — that is correct.
- `Written: /tmp/mock_delays/chart_ttft_delays.png` printed.
- `"TODO: chart_ttft_delays"` NOT printed.
- Open the PNG: two lines visible (H100 0s and H100 60s), both solid, different colors,
  single legend box, x-axis labels 1k–51k.

---

### Commit message
`Commit 6 — chart_ttft_delays`
