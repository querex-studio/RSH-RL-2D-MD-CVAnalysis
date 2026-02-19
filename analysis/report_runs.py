import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from analysis import run_utils


def find_runs(root):
    if not os.path.isdir(root):
        return []
    runs = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    return sorted(runs)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_explained(path):
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.size < 2:
        return float(arr[0]) if arr.size == 1 else None
    return float(arr[0] + arr[1])


def main():
    parser = argparse.ArgumentParser(description="Summarize analysis runs and generate KPI report.")
    parser.add_argument("--runs-root", default=None, help="Root folder that contains analysis_runs/")
    parser.add_argument("--out", default=None, help="Output report path (markdown)")
    args = parser.parse_args()

    runs_root = args.runs_root or getattr(config, "RUNS_DIR", "analysis_runs")
    if not os.path.isabs(runs_root):
        runs_root = os.path.join(ROOT_DIR, runs_root)

    runs = find_runs(runs_root)
    if not runs:
        raise FileNotFoundError("No analysis runs found.")

    if args.out:
        out_path = args.out
        report_dir = os.path.dirname(out_path)
    else:
        time_tag = run_utils.default_time_tag()
        report_dir = os.path.join("reports", time_tag)
        out_path = os.path.join(report_dir, "summary.md")
    os.makedirs(report_dir, exist_ok=True)

    rows = []
    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        meta = read_json(os.path.join(run_dir, "run.json")) or {}
        script = meta.get("script", "unknown")
        explained_path = os.path.join(run_dir, "data", "pca_explained.npy")
        pc12 = summarize_explained(explained_path)
        structures_dir = os.path.join(run_dir, "structures")
        struct_count = 0
        if os.path.isdir(structures_dir):
            for root, _, files in os.walk(structures_dir):
                struct_count += len([p for p in files if p.endswith(".pdb")])

        rows.append(
            {
                "run": run_name,
                "script": script,
                "pc1_pc2_cumulative": pc12,
                "structures": struct_count,
            }
        )

    # Write CSV summary
    csv_path = os.path.join(report_dir, "runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Report
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Analysis Runs Summary\n\n")
        f.write(f"Runs root: `{runs_root}`\n\n")
        f.write("## Per-run summary (see CSV)\n\n")
        f.write(f"- CSV: `{csv_path}`\n")


if __name__ == "__main__":
    main()
