# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

## Commit 7 — chart_ttft_delays dual legend

Target: replace the single `ax.legend()` call in `chart_ttft_delays` with two separate
legend boxes — one for GPU (line style) and one for delay value (color) — using
matplotlib proxy artists.

**CRITICAL CONSTRAINTS — do not deviate:**
- Touch ONLY the single `ax.legend()` call inside `chart_ttft_delays`. Every other
  line in the function stays exactly as-is. Do not touch any other function.
- Do not add any new imports. `matplotlib.lines.Line2D` is available as
  `matplotlib.lines.Line2D` — `matplotlib` is already imported at the top of the file.
- The plotting loop does NOT change. Labels on the plotted lines do not matter —
  the proxy artist approach builds both legends independently from scratch.
- GPU legend at `loc='upper left'`, delay legend at `loc='upper right'`.
  Both must be fully visible with no overlap.

---

### Steps

1. Open `benchmarks/visualize.py`. Find the single `ax.legend()` call inside
   `chart_ttft_delays`. That is the only line being replaced.

2. Replace `ax.legend()` with:

   ```python
   # GPU legend — black lines, varying line style
   gpu_handles = [
       matplotlib.lines.Line2D([0], [0], color="black", linestyle=GPU_LINESTYLES[gpu], label=GPU_LABELS[gpu])
       for gpu in GPUS
   ]
   gpu_leg = ax.legend(handles=gpu_handles, labels=[GPU_LABELS[gpu] for gpu in GPUS], loc="upper left")
   ax.add_artist(gpu_leg)

   # Delay legend — colored solid lines
   delay_handles = [
       matplotlib.lines.Line2D([0], [0], color=f"C{i}", linestyle="solid", label=f"{d}s")
       for i, d in enumerate(DELAYS)
   ]
   ax.legend(handles=delay_handles, labels=[f"{d}s" for d in DELAYS], loc="upper right")
   ```

   `GPU_LINESTYLES` and `DELAYS` are local variables already defined earlier in the
   function. They are in scope.

3. Smoke test — run:

   ```bash
   mkdir -p /tmp/mock_delays

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

4. Verify:
   - Script runs without error.
   - `Written: /tmp/mock_delays/chart_ttft_delays.png` is printed.
   - Open the PNG: two distinct legend boxes visible — upper-left for GPU (black lines,
     varying styles), upper-right for delay values (colored solid lines). No overlap.

---

### Commit message
`Commit 7 — chart_ttft_delays dual legend`
