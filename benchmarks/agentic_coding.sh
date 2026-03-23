#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/benchmark_lib.sh"

# Hardcoded ISL sweep constants — do not make configurable
ISL_VALUES=(1000 11000 21000 31000 41000 51000)
PREFIX_LENS=(0 1000 11000 21000 31000 41000)
NEW_TOKENS=(1000 10000 10000 10000 10000 10000)

# check_env_vars uses ${!var_name} (indirect expansion) which is incompatible with set -u;
# temporarily disable nounset so the function can detect and report missing vars itself.
set +u
check_env_vars MODEL API_URL TEST_TYPE RESULT_FILE
set -u

NUM_PROMPTS=${NUM_PROMPTS:-20}
DELAY_S=${DELAY_S:-0}

if [[ "$TEST_TYPE" != "ttft-caching" && "$TEST_TYPE" != "itl-bandwidth" && "$TEST_TYPE" != "ttft-delays" ]]; then
    echo "Error: TEST_TYPE must be one of: ttft-caching, itl-bandwidth, ttft-delays (got: '$TEST_TYPE')" >&2
    exit 1
fi

echo "agentic_coding.sh: validation complete. TEST_TYPE=$TEST_TYPE MODEL=$MODEL API_URL=$API_URL NUM_PROMPTS=$NUM_PROMPTS DELAY_S=$DELAY_S"
