# TODO.md
# Purpose: Current Git commit target (first line) and exact steps to achieve it.
# Updated as work progresses. If the original steps prove insufficient, add more.

---


Current target: Commit 9 — Mark all planning docs complete and record loose ends

## What this commit does

This is a pure documentation commit. No source code, no shell scripts, no Python,
no YAML. The only files touched are `.claude/PLAN.md`, `.claude/TODO.md`, and
`.claude/LOOSE_ENDS.md`.

Mark every commit in `PLAN.md` as ✓ DONE. Capture any remaining loose ends in
`LOOSE_ENDS.md`. Overwrite `TODO.md` with a "project complete" stub.

## Exact steps

1. **Update `PLAN.md`**: Mark Commit 9 itself as `~~...~~ ✓ DONE`. Every commit
   entry in the file must now have the strikethrough + checkmark format. Do not
   change any other content.

2. **Update `LOOSE_ENDS.md`**: Review the entire implementation (Commits 1–8) and
   add any loose ends not yet recorded. At minimum, verify the existing entry about
   `check_env_vars` / `set -u` incompatibility is still accurate. Add new entries
   if anything was noticed but not yet written down. Do not remove existing entries.

3. **Overwrite `TODO.md`** with a stub that says the project is complete and points
   to `PLAN.md` for history and `LOOSE_ENDS.md` for known issues.

4. **Run the test suite one final time**: `cd utils && python -m pytest matrix_logic/ -v`
   All 142 tests must pass.

5. Done. Report back.

## Hard constraints

- Touch ONLY `.claude/PLAN.md`, `.claude/TODO.md`, `.claude/LOOSE_ENDS.md`.
- Zero changes to any source file (no `.sh`, `.py`, `.yaml`, or other file).
- Do NOT reformat or restructure `PLAN.md` — only update the Commit 9 entry.
- Do NOT invent loose ends. Only record things actually observed during implementation.
- Do NOT add a `CHANGELOG`, `RELEASE`, or any other new file.
