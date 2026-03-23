# SPEC.md
# Purpose: Precise technical specification for the agentic coding benchmark results storage
# and visualization feature. Written 2026-03-22. See OVERVIEW.md for context and motivation.

---

## GPU Lineup

Three GPU types. B300 is excluded — no AWS instance type available.

| GPU  | AWS Instance Type  | Node Config  | HBM per Node |
|------|--------------------|--------------|--------------|
| H100 | `p5.48xlarge`      | 8×H100 SXM   | 640 GB       |
| H200 | `p5en.48xlarge`    | 8×H200 SXM   | 1,128 GB     |
| B200 | `p6-b200.48xlarge` | 8×B200 SXM   | 1,536 GB     |

---

## AWS Instance Provisioning

This section specifies exactly how to bring up each GPU instance so that
`agentic_coding.sh` can run against it.

### Step 1 — Launch the instance

In the AWS console (or via `aws ec2 run-instances`), launch the appropriate instance type
from the table above. Use the following settings:

- **AMI**: AWS Deep Learning AMI (DLAMI) — select the latest DLAMI for Ubuntu 22.04 with
  CUDA 12.x (search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" in the AMI catalog).
  The DLAMI provides pre-installed NVIDIA drivers; no manual driver installation is needed.
- **Storage**: at least 200 GB on the root EBS volume (the model weights are ~120 GB).
- **Security group**: open **inbound TCP port 8000** from your local IP (or `0.0.0.0/0`
  for simplicity during testing). This is the port vLLM will serve on.
- **Key pair**: attach a key pair you have locally so you can SSH in.

### Step 2 — SSH into the instance

```bash
ssh -i /path/to/key.pem ubuntu@<instance-public-ip>
```

### Step 3 — Pull the model weights

The model is `openai/gpt-oss-120b` on HuggingFace. Pull it before starting the server so
the Docker container does not time out downloading on first request.

```bash
pip install huggingface_hub
huggingface-cli download openai/gpt-oss-120b --local-dir ~/models/gpt-oss-120b
```

If the model is gated, first authenticate:

```bash
huggingface-cli login
```

### Step 4 — Start the vLLM inference server

Run the following Docker command. It starts an OpenAI-compatible HTTP server on port 8000.

```bash
docker run --rm --gpus all \
    -v ~/models/gpt-oss-120b:/model \
    -p 8000:8000 \
    vllm/vllm-openai:v0.9.0 \
    --model /model \
    --tensor-parallel-size 8 \
    --dtype auto \
    --max-model-len 60000 \
    --served-model-name openai/gpt-oss-120b
```

Flag explanations:
- `--gpus all`: expose all 8 GPUs to the container
- `-v ~/models/gpt-oss-120b:/model`: mount the downloaded weights at `/model` inside the container
- `-p 8000:8000`: publish port 8000 to the host
- `--tensor-parallel-size 8`: shard the model across all 8 GPUs
- `--dtype auto`: let vLLM pick the optimal dtype (will use the model's native MXFP4/BF16)
- `--max-model-len 60000`: sufficient headroom above the 51k max ISL used in the benchmarks
- `--served-model-name openai/gpt-oss-120b`: the model name string clients must use in requests

The server is ready when the log line `INFO: Application startup complete.` appears.

### Step 5 — Verify the server is up

From the instance (or from your local machine if port 8000 is open):

```bash
curl http://localhost:8000/v1/models
```

Expected response: a JSON object listing `openai/gpt-oss-120b` as an available model.

### Exposed interface

The vLLM server exposes an **OpenAI-compatible REST API** on `http://<instance-public-ip>:8000`.

The environment variables required by `agentic_coding.sh` map as follows:

| Variable      | Value                                      |
|---------------|--------------------------------------------|
| `API_URL`     | `http://<instance-public-ip>:8000`         |
| `MODEL`       | `openai/gpt-oss-120b`                      |
| `TEST_TYPE`   | one of `ttft-caching`, `itl-bandwidth`, `ttft-delays` |
| `RESULT_FILE` | path on the instance where raw JSON is written |

No API key is required (vLLM's default configuration does not enforce authentication).

---

## Scope

Two new Python scripts:

1. `benchmarks/wrap_result.py` — wraps raw `agentic_coding.sh` output with metadata
2. `benchmarks/visualize.py` — reads a results directory and produces three PNG charts

No other files are modified. `.gitignore` is not touched. `agentic_coding.sh`,
`benchmark_lib.sh`, and all Pydantic validation code are untouched.

---

## Results Directory Layout

```
benchmarks/results/
└── YYYY-MM-DDTHH:MM:SS/          ← one subdir per suite run, created manually
    ├── h100_ttft-caching.json
    ├── h100_itl-bandwidth.json
    ├── h100_ttft-delays_0s.json
    ├── h100_ttft-delays_1s.json
    ├── h100_ttft-delays_5s.json
    ├── h100_ttft-delays_10s.json
    ├── h100_ttft-delays_60s.json
    ├── h200_ttft-caching.json
    ├── h200_itl-bandwidth.json
    ├── h200_ttft-delays_0s.json
    ├── h200_ttft-delays_1s.json
    ├── h200_ttft-delays_5s.json
    ├── h200_ttft-delays_10s.json
    ├── h200_ttft-delays_60s.json
    ├── b200_ttft-caching.json
    ├── b200_itl-bandwidth.json
    ├── b200_ttft-delays_0s.json
    ├── b200_ttft-delays_1s.json
    ├── b200_ttft-delays_5s.json
    ├── b200_ttft-delays_10s.json
    ├── b200_ttft-delays_60s.json
    ├── chart_ttft_caching.png     ← written by visualize.py
    ├── chart_itl_bandwidth.png
    └── chart_ttft_delays.png
```

The developer creates the timestamped subdirectory manually before running any scripts.

---

## JSON Schema

Every result file has this exact structure:

```json
{
  "meta": {
    "gpu": "h100",
    "test_type": "ttft-caching",
    "delay_s": 0,
    "model": "openai/gpt-oss-120b",
    "num_prompts": 20,
    "timestamp": "2026-03-22T14:30:00"
  },
  "turns": [
    {
      "turn": 0,
      "isl": 1000,
      "prefix_len": 0,
      "new_tokens": 1000,
      "delay_s": 0,
      "ttft_mean": 123.4,
      "ttft_p50": 120.1,
      "ttft_p99": 145.2,
      "itl_mean": 10.1,
      "itl_p50": 9.8,
      "itl_p99": 12.3,
      "num_prompts": 20
    },
    ...5 more turns...
  ]
}
```

Field constraints:
- `meta.gpu`: one of `"h100"`, `"h200"`, `"b200"`
- `meta.test_type`: one of `"ttft-caching"`, `"itl-bandwidth"`, `"ttft-delays"`
- `meta.delay_s`: integer; 0 for `ttft-caching` and `itl-bandwidth`, one of {0,1,5,10,60} for `ttft-delays`
- `meta.model`: string, e.g. `"openai/gpt-oss-120b"`
- `meta.num_prompts`: integer
- `meta.timestamp`: ISO 8601 string, e.g. `"2026-03-22T14:30:00"`
- `turns`: array of exactly 6 objects
- `turns[i].turn`: integer 0–5
- `turns[i].isl`: one of {1000, 11000, 21000, 31000, 41000, 51000}
- All latency fields (`ttft_mean`, `ttft_p50`, `ttft_p99`, `itl_mean`, `itl_p50`, `itl_p99`): float (ms) or null if not measured

---

## wrap_result.py

### Location
`benchmarks/wrap_result.py`

### Purpose
Takes the raw JSON array written by `agentic_coding.sh` (a list of 6 turn objects) and
wraps it with a `meta` block to produce the canonical result file format.

### CLI interface

```bash
python benchmarks/wrap_result.py \
    --input PATH \
    --output PATH \
    --gpu {h100,h200,b200,b300} \
    --test-type {ttft-caching,itl-bandwidth,ttft-delays} \
    --delay-s INT \
    --model STRING \
    --num-prompts INT \
    --timestamp STRING
```

All eight arguments are required. No defaults.

| Argument | Type | Description |
|---|---|---|
| `--input` | path | Path to raw JSON file output by `agentic_coding.sh` |
| `--output` | path | Destination path for the wrapped result file |
| `--gpu` | str | GPU identifier: `h100`, `h200`, or `b200` |
| `--test-type` | str | Test type string |
| `--delay-s` | int | Delay in seconds (0 for non-delay tests) |
| `--model` | str | Model name string |
| `--num-prompts` | int | Number of prompts used in the run |
| `--timestamp` | str | ISO 8601 timestamp string for this suite run |

### Behavior

1. Read and parse `--input` as JSON. If the file does not exist or is not valid JSON, print
   an error to stderr and exit 1.
2. Validate that the parsed value is a list of exactly 6 elements. If not, print an error
   to stderr and exit 1.
3. Validate that `--gpu` is one of `{h100, h200, b200}`. If not, print an error to
   stderr and exit 1.
4. Validate that `--test-type` is one of `{ttft-caching, itl-bandwidth, ttft-delays}`. If
   not, print an error to stderr and exit 1.
5. Check that the parent directory of `--output` exists. If not, print an error to stderr
   and exit 1. Do NOT create missing directories.
6. Construct the output dict: `{"meta": {...}, "turns": <input list>}`.
7. Write to `--output` with `json.dump(..., indent=2)`.
8. Print one confirmation line to stdout: `Written: <output path>`.
9. Exit 0.

### Dependencies
Standard library only: `argparse`, `json`, `sys`, `os`.

---

## visualize.py

### Location
`benchmarks/visualize.py`

### Purpose
Reads all result JSON files from a results subdirectory and produces three PNG charts.

### CLI interface

```bash
python benchmarks/visualize.py RESULTS_DIR
```

Single positional argument: path to a timestamped results subdirectory.

### Behavior

1. Verify `RESULTS_DIR` exists and is a directory. If not, print an error to stderr and
   exit 1.
2. For each of the three charts, load the relevant JSON files from `RESULTS_DIR`.
3. If a file is missing for a particular GPU, print a warning to stdout
   (`Warning: missing <filename>, skipping <GPU>`) and skip that GPU — do not crash.
4. If NO files are found for a chart (all four GPUs missing), print a warning and skip
   that chart entirely — do not crash.
5. Generate the charts and write them to `RESULTS_DIR` as PNG files.
6. Print one line per chart written: `Written: <path>`.
7. Exit 0.

### File loading logic

**Chart 1 (ttft-caching):** load `{gpu}_ttft-caching.json` for each GPU.
**Chart 2 (itl-bandwidth):** load `{gpu}_itl-bandwidth.json` for each GPU.
**Chart 3 (ttft-delays):** load `{gpu}_ttft-delays_{d}s.json` for each GPU and each delay
value d ∈ {0, 1, 5, 10, 60}.

### Chart 1 — `chart_ttft_caching.png`

| Property | Value |
|---|---|
| Title | `"TTFT vs. Context Length (KV Cache Reuse)"` |
| X-axis label | `"Input Sequence Length (tokens)"` |
| X-axis ticks | positions: [1000,11000,21000,31000,41000,51000]; labels: ["1k","11k","21k","31k","41k","51k"] |
| Y-axis label | `"Mean TTFT (ms)"` |
| Lines | one per GPU; x = isl values, y = `ttft_mean` values from turns array |
| Colors | matplotlib tab10 defaults, in GPU order: H100, H200, B200 |
| Line style | solid for all |
| Legend | single legend, labels: `"H100"`, `"H200"`, `"B200"` |
| Figure size | 10×6 inches |
| DPI | 150 |

### Chart 2 — `chart_itl_bandwidth.png`

| Property | Value |
|---|---|
| Title | `"ITL vs. Context Length (Memory Bandwidth)"` |
| X-axis label | `"Input Sequence Length (tokens)"` |
| X-axis ticks | same as Chart 1 |
| Y-axis label | `"Mean ITL (ms)"` |
| Lines | one per GPU; x = isl values, y = `itl_mean` values from turns array |
| Colors | matplotlib tab10 defaults, same GPU order |
| Line style | solid for all |
| Legend | single legend, labels: `"H100"`, `"H200"`, `"B200"`, `"B300"` |
| Figure size | 10×6 inches |
| DPI | 150 |

### Chart 3 — `chart_ttft_delays.png`

| Property | Value |
|---|---|
| Title | `"TTFT vs. Context Length (Cache Eviction Under Delays)"` |
| X-axis label | `"Input Sequence Length (tokens)"` |
| X-axis ticks | same as Chart 1 |
| Y-axis label | `"Mean TTFT (ms)"` |
| Lines | one per (GPU, delay) combination; x = isl values, y = `ttft_mean` values |
| Colors | tab10 defaults by delay value, in order: 0s, 1s, 5s, 10s, 60s |
| Line styles | by GPU: H100=solid, H200=dashed, B200=dotted |
| Legends | **two separate legends**: one for GPU (line style), one for delay value (color) |
| GPU legend | labels: `"H100"`, `"H200"`, `"B200"` with their line styles, black color |
| Delay legend | labels: `"0s"`, `"1s"`, `"5s"`, `"10s"`, `"60s"` with their colors, solid line style |
| Figure size | 10×6 inches |
| DPI | 150 |

### Two-legend implementation note

Use `matplotlib.legend.Legend` directly. Create the GPU legend first with
`ax.legend(gpu_handles, gpu_labels, ...)`, then add the delay legend as a second artist
with `ax.add_artist(...)`. Position the GPU legend at `loc='upper left'` and the delay
legend at `loc='upper right'` (or wherever they don't overlap — adjust to taste, but
both must be visible without overlap).

### Dependencies
`matplotlib` only (plus standard library: `argparse`, `json`, `os`, `sys`).

---

## What Is Not In Scope

- No changes to `agentic_coding.sh`, `benchmark_lib.sh`, or any validation code
- No `.gitignore` modifications
- No automated SSH orchestration
- No Makefile or shell wrapper
- No interactive charts
- No validation in `wrap_result.py` beyond: valid JSON, list of 6 elements, valid gpu/test-type strings, output parent dir exists
- No validation in `visualize.py` beyond: directory exists, files present
