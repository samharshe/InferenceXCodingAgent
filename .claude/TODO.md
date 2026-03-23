# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---

Current target: Commit 1 — Add `AgenticSearchSpaceEntry` and `AgenticMasterConfigEntry` to `validation.py`

Steps:
1. Open `utils/matrix_logic/validation.py`.
2. After the existing `MultiNodeSearchSpaceEntry` class, add `AgenticSearchSpaceEntry`:
   - Fields: `tp: int`, `ep: Optional[int] = None`, `dp_attn: Optional[bool] = Field(default=None, alias='dp-attn')`
   - `model_config = ConfigDict(extra='forbid', populate_by_name=True)`
3. After the existing `MultiNodeMasterConfigEntry` class, add `AgenticMasterConfigEntry`:
   - Fields: `image`, `model`, `model_prefix` (alias `model-prefix`), `precision`, `framework`, `runner`
   - `multinode: Literal[False]`, `agentic: Literal[True]`
   - `test_type: Literal['ttft-caching', 'itl-bandwidth', 'ttft-delays']` (alias `test-type`)
   - `num_prompts: int = Field(default=20, alias='num-prompts')`
   - `delays: Optional[List[int]] = None`
   - `search_space: List[AgenticSearchSpaceEntry]` (alias `search-space`)
   - `@model_validator(mode='after')` that enforces delays ↔ test_type constraint
4. Do NOT touch any existing class.
5. Run `cd utils && python -m pytest matrix_logic/ -v` — confirm all existing tests still pass.
6. Commit.
