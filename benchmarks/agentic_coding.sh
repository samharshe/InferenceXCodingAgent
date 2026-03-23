#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/benchmark_lib.sh"

# Hardcoded ISL sweep constants — do not make configurable
ISL_VALUES=(1000 11000 21000 31000 41000 51000)
PREFIX_LENS=(0 1000 11000 21000 31000 41000)
NEW_TOKENS=(1000 10000 10000 10000 10000 10000)

check_env_vars MODEL API_URL TEST_TYPE RESULT_FILE

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

# Verified output keys from benchmark_serving.py (default --percentile-metrics=ttft,tpot,itl
# and --metric-percentiles=99): mean_ttft_ms, median_ttft_ms (p50), p99_ttft_ms,
# mean_itl_ms, median_itl_ms (p50), p99_itl_ms.
# Aggregate per-turn results into $RESULT_FILE
python3 -c "
import json, sys

ISL_VALUES = [1000, 11000, 21000, 31000, 41000, 51000]
PREFIX_LENS = [0, 1000, 11000, 21000, 31000, 41000]
NEW_TOKENS  = [1000, 10000, 10000, 10000, 10000, 10000]

turns = []
for t in range(6):
    path = f'/tmp/agentic_turn_{t}.json'
    with open(path) as f:
        d = json.load(f)
    turns.append({
        'turn':       t,
        'isl':        ISL_VALUES[t],
        'prefix_len': PREFIX_LENS[t],
        'new_tokens': NEW_TOKENS[t],
        'delay_s':    int('${DELAY_S}'),
        'ttft_mean':  d.get('mean_ttft_ms'),
        'ttft_p50':   d.get('median_ttft_ms'),
        'ttft_p99':   d.get('p99_ttft_ms'),
        'itl_mean':   d.get('mean_itl_ms'),
        'itl_p50':    d.get('median_itl_ms'),
        'itl_p99':    d.get('p99_itl_ms'),
        'num_prompts': int('${NUM_PROMPTS}'),
    })

with open('${RESULT_FILE}', 'w') as f:
    json.dump(turns, f, indent=2)
"

# Clean up per-turn temp files
for t in 0 1 2 3 4 5; do
    rm -f "/tmp/agentic_turn_${t}.json"
done

echo "Results written to $RESULT_FILE."
