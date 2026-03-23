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

for t in 0 1 2 3 4 5; do
    if [[ $t -gt 0 && "${DELAY_S}" -gt 0 ]]; then
        sleep "$DELAY_S"
    fi

    TURN_RESULT_FILE="/tmp/agentic_turn_${t}.json"

    # Parse host and port from API_URL (e.g. http://hostname:8000)
    API_HOST=$(echo "$API_URL" | sed 's|https\?://||' | cut -d: -f1)
    API_PORT=$(echo "$API_URL" | sed 's|.*:||')

    python3 "$(dirname "$0")/../utils/bench_serving/benchmark_serving.py" \
        --backend openai \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "${NEW_TOKENS[$t]}" \
        --random-prefix-len "${PREFIX_LENS[$t]}" \
        --random-output-len 1000 \
        --random-range-ratio 1.0 \
        --num-prompts "$NUM_PROMPTS" \
        --save-result \
        --result-filename "$TURN_RESULT_FILE"

    echo "Turn $t complete: ISL=${ISL_VALUES[$t]}"
done

echo "All turns complete."
