# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

Current target: Commit 4 — Add agentic config entries to `nvidia-master.yaml`

## What this commit does

Append three placeholder agentic entries to `.github/configs/nvidia-master.yaml`. Nothing else.

No changes to Python files. No new files. No touching existing YAML entries.

## Exact steps

1. Open `.github/configs/nvidia-master.yaml`. Scroll to the very end of the file.
2. Append these three entries exactly as written (use literal `<placeholder>` strings for image and model — do NOT invent real values):

```yaml
gptoss-fp8-b200-vllm-agentic-ttft-caching:
  image: <placeholder>  # TODO: replace with real values
  model: <placeholder>  # TODO: replace with real values
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
  image: <placeholder>  # TODO: replace with real values
  model: <placeholder>  # TODO: replace with real values
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
  image: <placeholder>  # TODO: replace with real values
  model: <placeholder>  # TODO: replace with real values
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

3. Run `python utils/matrix_logic/generate_sweep_configs.py full-sweep --master-config .github/configs/nvidia-master.yaml` from the repo root. It must NOT raise any validation errors. (The generate logic will skip the new entries because they have no `seq-len-configs` — this is expected and acceptable. We only care that `validate_master_config` does not blow up.)
4. Run `cd utils && python -m pytest matrix_logic/ -v` — all 142 tests must still pass.
5. Done. Report back. Do not proceed to Commit 5.

## Hard constraints

- Do NOT modify `validation.py` or any test file.
- Do NOT modify any existing YAML entry.
- Do NOT create any new files.
- The YAML keys must be exactly as written above: `gptoss-fp8-b200-vllm-agentic-ttft-caching`, `gptoss-fp8-b200-vllm-agentic-itl-bandwidth`, `gptoss-fp8-b200-vllm-agentic-ttft-delays`.
- Use literal `<placeholder>` for `image` and `model` — do not invent values.
- YAML field names must be kebab-case (e.g. `model-prefix`, `test-type`, `num-prompts`, `search-space`) — this is what `AgenticMasterConfigEntry` expects via its field aliases.
