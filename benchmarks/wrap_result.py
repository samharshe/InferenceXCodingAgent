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
    print(args)


if __name__ == "__main__":
    main()
