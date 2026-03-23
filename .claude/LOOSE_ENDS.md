# LOOSE_ENDS.md
# Purpose: Potential bugs, refactors, or improvements noticed along the way
# that are NOT part of the current development goal. Captured here so they aren't forgotten
# but don't distract from the task at hand.

---

## `check_env_vars` incompatible with `set -u`

`benchmark_lib.sh`'s `check_env_vars` uses `${!var_name}` (indirect variable expansion) to test
whether each named variable is set. With `set -u` (nounset) active, bash raises an "unbound
variable" error on the indirect expansion itself before the function can detect and report the
missing variable.

**Current workaround** (`benchmarks/agentic_coding.sh`): temporarily disable nounset with
`set +u` before calling `check_env_vars`, then re-enable with `set -u` immediately after.

**Cleaner fix (not done)**: Update `check_env_vars` in `benchmark_lib.sh` to use a nounset-safe
check, e.g. `${!var_name+x}` (parameter expansion that returns `x` if set, empty if unset) —
this works under `set -u` without triggering an error. Requires touching `benchmark_lib.sh`,
which was out of scope for Commit 5.

**Resolved in Commit 10.**
