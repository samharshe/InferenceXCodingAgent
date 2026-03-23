import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Wrap raw agentic_coding.sh JSON output with metadata."
    )
    parser.add_argument("--input", required=True, help="Path to raw JSON file")
    parser.add_argument("--output", required=True, help="Destination path for wrapped result")
    parser.add_argument("--gpu", required=True, help="GPU identifier: h100, h200, or b200")
    parser.add_argument("--test-type", required=True, help="Test type: ttft-caching, itl-bandwidth, or ttft-delays")
    parser.add_argument("--delay-s", required=True, type=int, help="Delay in seconds (0 for non-delay tests)")
    parser.add_argument("--model", required=True, help="Model name string")
    parser.add_argument("--num-prompts", required=True, type=int, help="Number of prompts used in the run")
    parser.add_argument("--timestamp", required=True, help="ISO 8601 timestamp string")

    args = parser.parse_args()

    # Read and parse input
    try:
        with open(args.input) as f:
            turns = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate turns count
    if not isinstance(turns, list) or len(turns) != 6:
        print(f"Error: input must be a JSON array of exactly 6 elements, got {type(turns).__name__} of length {len(turns) if isinstance(turns, list) else 'N/A'}", file=sys.stderr)
        sys.exit(1)

    # Validate gpu
    valid_gpus = {"h100", "h200", "b200"}
    if args.gpu not in valid_gpus:
        print(f"Error: --gpu must be one of {sorted(valid_gpus)}, got: {args.gpu!r}", file=sys.stderr)
        sys.exit(1)

    # Validate test-type
    valid_test_types = {"ttft-caching", "itl-bandwidth", "ttft-delays"}
    if args.test_type not in valid_test_types:
        print(f"Error: --test-type must be one of {sorted(valid_test_types)}, got: {args.test_type!r}", file=sys.stderr)
        sys.exit(1)

    # Check output parent directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(output_dir):
        print(f"Error: output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Build and write output
    result = {
        "meta": {
            "gpu": args.gpu,
            "test_type": args.test_type,
            "delay_s": args.delay_s,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "timestamp": args.timestamp,
        },
        "turns": turns,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
