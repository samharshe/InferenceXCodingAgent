import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


GPUS = ["h100", "h200", "b200"]
GPU_LABELS = {"h100": "H100", "h200": "H200", "b200": "B200"}
ISL_POSITIONS = [1000, 11000, 21000, 31000, 41000, 51000]
ISL_LABELS = ["1k", "11k", "21k", "31k", "41k", "51k"]


def chart_ttft_caching(results_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False

    for gpu in GPUS:
        data = load_file(results_dir, f"{gpu}_ttft-caching.json")
        if data is None:
            continue
        isl, ttft = get_isl_and_metric(data["turns"], "ttft_mean")
        ax.plot(isl, ttft, label=GPU_LABELS[gpu])
        any_data = True

    if not any_data:
        print("Warning: no data for chart_ttft_caching, skipping")
        plt.close(fig)
        return

    ax.set_xticks(ISL_POSITIONS)
    ax.set_xticklabels(ISL_LABELS)
    ax.set_title("TTFT vs. Context Length (KV Cache Reuse)")
    ax.set_xlabel("Input Sequence Length (tokens)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.legend()

    out = os.path.join(results_dir, "chart_ttft_caching.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Written: {out}")


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
