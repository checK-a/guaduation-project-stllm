import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_RE = re.compile(
    r"(?P<dataset>(?:ili_us_states_h\d+|us_states_nhsn_flu_hosp_h\d+)(?:_leakfree)?)"
)


def infer_dataset_from_path(path):
    match = DATASET_RE.search(str(path))
    return match.group("dataset") if match else "unknown"


def infer_run_name(path):
    return path.parent.name


def write_markdown(df, out_md):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Conformal Prediction Interval Results\n\n")
        f.write("Intervals are calibrated on validation residuals and evaluated on the test split.\n\n")
        rounded = df.round(6)
        columns = list(rounded.columns)
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for _, row in rounded.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Collect conformal interval CSVs from training logs.")
    parser.add_argument("--root", type=str, default="logs", help="root directory to scan")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/conformal_intervals.csv",
        help="glob pattern under root",
    )
    parser.add_argument("--out_csv", type=str, default="review/conformal_interval_results.csv")
    parser.add_argument("--out_md", type=str, default="review/conformal_interval_results.md")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob(args.pattern))
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df.insert(0, "run", infer_run_name(path))
        df.insert(0, "dataset", infer_dataset_from_path(path))
        df.insert(0, "path", str(path))
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No conformal interval CSVs found under {root} with pattern {args.pattern}")

    out = pd.concat(rows, ignore_index=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_markdown(out, Path(args.out_md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
