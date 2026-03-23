# PLAN.md
# Purpose: Step-by-step implementation plan for wrap_result.py and visualize.py.
# Spec: .claude/SPEC.md. Written 2026-03-22.

---

## Overview

Two new files to implement. Nothing else is touched.

| File | Status |
|------|--------|
| `benchmarks/wrap_result.py` | New — wraps raw JSON with meta block |
| `benchmarks/visualize.py` | New — reads results dir, writes 3 PNGs |

---

## ~~Phase 1 — wrap_result.py (Day 1, Morning)~~ DONE

### ~~Commit 1 — argparse skeleton~~ DONE

~~Create `benchmarks/wrap_result.py` with all 8 required CLI args and a `main()` that
parses and prints them. No logic yet.~~

```
--input PATH
--output PATH
--gpu STR
--test-type STR
--delay-s INT
--model STR
--num-prompts INT
--timestamp STR
```

~~**Smoke test:** `python benchmarks/wrap_result.py --help` shows all 8 args.~~

### ~~Commit 2 — full implementation~~ DONE

~~Add all validation and output logic:~~

~~1. Read `--input` as JSON; stderr + exit 1 if file missing or invalid JSON.~~
~~2. Validate parsed value is list of exactly 6 elements; stderr + exit 1 if not.~~
~~3. Validate `--gpu` ∈ `{h100, h200, b200}`; stderr + exit 1 if not.~~
~~4. Validate `--test-type` ∈ `{ttft-caching, itl-bandwidth, ttft-delays}`; stderr + exit 1 if not.~~
~~5. Check parent dir of `--output` exists; stderr + exit 1 if not (do NOT mkdir).~~
~~6. Write `{"meta": {...}, "turns": <input list>}` with `json.dump(..., indent=2)`.~~
~~7. Print `Written: <output path>`; exit 0.~~

~~Dependencies: `argparse`, `json`, `sys`, `os` only.~~

~~**Smoke test:** run against a hand-crafted 6-element JSON array; verify output file has
correct `meta` block and `turns` array matches input.~~

---

## Phase 2 — visualize.py (Day 1, Afternoon + Day 2, Morning)

### ~~Commit 3 — skeleton with data loader~~ DONE

~~Create `benchmarks/visualize.py` with:~~

~~- Single positional arg `RESULTS_DIR`.~~
~~- Directory existence check; stderr + exit 1 if not a directory.~~
~~- `load_file(results_dir, filename)` helper: reads and returns parsed JSON, or `None`
  with printed warning if file is missing.~~
~~- `get_isl_and_metric(turns, metric_key)` helper: returns `(isl_list, metric_list)`.~~
~~- Three stub chart functions: `chart_ttft_caching`, `chart_itl_bandwidth`,
  `chart_ttft_delays` — each just prints `"TODO: <name>"` and returns.~~
~~- `main()` calls all three stubs.~~

~~**Smoke test:** `python benchmarks/visualize.py /tmp/` runs without crash, prints three
TODO lines.~~

### ~~Commit 4 — chart_ttft_caching~~ DONE

~~Implement `chart_ttft_caching(results_dir)`:~~

~~- GPUs: `[h100, h200, b200]`. Load `{gpu}_ttft-caching.json` for each.~~
~~- Skip GPU with warning if file missing. Skip chart entirely (with warning) if all missing.~~
~~- Plot one line per GPU: x = isl, y = `ttft_mean`.~~
~~- X-ticks: positions `[1000,11000,21000,31000,41000,51000]`, labels `["1k","11k","21k","31k","41k","51k"]`.~~
~~- Title: `"TTFT vs. Context Length (KV Cache Reuse)"`.~~
~~- X-label: `"Input Sequence Length (tokens)"`, Y-label: `"Mean TTFT (ms)"`.~~
~~- Legend labels: `"H100"`, `"H200"`, `"B200"`.~~
~~- Figure: 10×6 inches, 150 DPI. Save to `{results_dir}/chart_ttft_caching.png`.~~
~~- Print `Written: <path>`.~~

~~**Smoke test:** create a mock results dir with one `h100_ttft-caching.json`, run
visualize.py, open PNG and verify it renders.~~

### Commit 5 — chart_itl_bandwidth

Implement `chart_itl_bandwidth(results_dir)`. Identical structure to chart 1 except:

- Loads `{gpu}_itl-bandwidth.json`.
- Y metric: `itl_mean`.
- Title: `"ITL vs. Context Length (Memory Bandwidth)"`.
- Y-label: `"Mean ITL (ms)"`.
- Output: `chart_itl_bandwidth.png`.

**Smoke test:** same mock dir, verify PNG renders.

### Commit 6 — chart_ttft_delays (lines only, single legend)

Implement `chart_ttft_delays(results_dir)`:

- Delays: `[0, 1, 5, 10, 60]`. GPUs: `[h100, h200, b200]`.
- Load `{gpu}_ttft-delays_{d}s.json` for each (GPU, delay) pair. Skip missing with warning.
- Line style per GPU: H100=solid, H200=dashed, B200=dotted.
- Color per delay: tab10 C0–C4 for 0s, 1s, 5s, 10s, 60s.
- Title: `"TTFT vs. Context Length (Cache Eviction Under Delays)"`.
- X/Y axes same as other charts. Single combined legend for now.
- Output: `chart_ttft_delays.png`. Print `Written: <path>`.

**Smoke test:** mock dir with a few delay files, verify PNG renders with multiple lines.

### Commit 7 — chart_ttft_delays dual legend

Replace single legend with two per spec, using `matplotlib.lines.Line2D` proxy artists:

**GPU legend** — black lines varying line style, `loc='upper left'`:
```python
gpu_leg = ax.legend(gpu_handles, gpu_labels, loc='upper left')
ax.add_artist(gpu_leg)
```

**Delay legend** — colored solid lines, `loc='upper right'`:
```python
ax.legend(delay_handles, delay_labels, loc='upper right')
```

Both must be visible without overlap.

**Smoke test:** verify the PNG shows two distinct labeled legend boxes.

---

## Verification (End of Phase 2)

```bash
# 1. Create mock raw input
python -c "
import json
turns = [{'turn': i, 'isl': 1000+i*10000, 'prefix_len': i*1000, 'new_tokens': 1000,
          'delay_s': 0, 'ttft_mean': 100.0+i*5, 'ttft_p50': 98.0, 'ttft_p99': 120.0,
          'itl_mean': 10.0+i*0.5, 'itl_p50': 9.8, 'itl_p99': 12.0, 'num_prompts': 20}
         for i in range(6)]
print(json.dumps(turns))
" > /tmp/mock_raw.json

# 2. Create results dir and wrap
mkdir -p benchmarks/results/2026-03-22T00:00:00
python benchmarks/wrap_result.py \
  --input /tmp/mock_raw.json \
  --output benchmarks/results/2026-03-22T00:00:00/h100_ttft-caching.json \
  --gpu h100 --test-type ttft-caching --delay-s 0 \
  --model openai/gpt-oss-120b --num-prompts 20 \
  --timestamp 2026-03-22T00:00:00

# 3. Visualize
python benchmarks/visualize.py benchmarks/results/2026-03-22T00:00:00/

# 4. Inspect PNGs
open benchmarks/results/2026-03-22T00:00:00/chart_ttft_caching.png
open benchmarks/results/2026-03-22T00:00:00/chart_itl_bandwidth.png
open benchmarks/results/2026-03-22T00:00:00/chart_ttft_delays.png
```
