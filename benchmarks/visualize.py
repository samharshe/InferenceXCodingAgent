import argparse
import json
import os
import sys


def load_file(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.isfile(path):
        print(f"Warning: missing {filename}, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def get_isl_and_metric(turns, metric_key):
    isl = [t["isl"] for t in turns]
    values = [t.get(metric_key) for t in turns]
    return isl, values


def chart_ttft_caching(results_dir):
    print("TODO: chart_ttft_caching")


def chart_itl_bandwidth(results_dir):
    print("TODO: chart_itl_bandwidth")


def chart_ttft_delays(results_dir):
    print("TODO: chart_ttft_delays")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from a results directory."
    )
    parser.add_argument("results_dir", help="Path to timestamped results subdirectory")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: not a directory: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    chart_ttft_caching(args.results_dir)
    chart_itl_bandwidth(args.results_dir)
    chart_ttft_delays(args.results_dir)


if __name__ == "__main__":
    main()
