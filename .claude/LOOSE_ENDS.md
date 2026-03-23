# LOOSE_ENDS.md
# Purpose: Potential bugs, refactors, or improvements noticed along the way
# that are NOT part of the current development goal. Captured here so they aren't forgotten
# but don't distract from the task at hand.

---

## `generate_sweep_configs.py` crashes on agentic entries

`generate_sweep_configs.py full-sweep` crashes with `KeyError: 'seq-len-configs'` when it reaches the three new agentic entries in `nvidia-master.yaml`. The script unconditionally accesses `val[Fields.SEQ_LEN_CONFIGS.value]` at line 160 with no skip logic for entries that lack that field.

`validate_master_config` (the Pydantic layer) works correctly and does not blow up. The crash is downstream in the generate loop.

**Fix needed:** Add an `is_agentic` skip/continue guard in `generate_full_sweep` (around line 160 of `generate_sweep_configs.py`) before accessing `seq-len-configs`. Something like:
```python
if val.get('agentic', False):
    continue
```
This was intentionally deferred — Commit 4 constraints prohibited Python file changes.
