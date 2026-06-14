import argparse
import re
from pathlib import Path

import pandas as pd


HORIZON_RE = re.compile(
    r"Evaluate .* on test data for horizon (?P<horizon>\d+), "
    r"Test MAE: (?P<mae>[-+0-9.eE]+), "
    r"Test RMSE: (?P<rmse>[-+0-9.eE]+), "
    r"Test MAPE: (?P<mape>[-+0-9.eE]+), "
    r"Test WMAPE: (?P<wmape>[-+0-9.eE]+)"
)
AVG_RE = re.compile(
    r"On average over (?P<num_horizons>\d+) horizons, "
    r"Test MAE: (?P<mae>[-+0-9.eE]+), "
    r"Test RMSE: (?P<rmse>[-+0-9.eE]+), "
    r"Test MAPE: (?P<mape>[-+0-9.eE]+), "
    r"Test WMAPE: (?P<wmape>[-+0-9.eE]+)"
)


def parse_log_name(path):
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        raise ValueError(f"Cannot parse seed from log name: {path.name}")
    dataset, rest = stem.split(marker, 1)
    seed_text, graph = rest.split("_", 1)
    return dataset, int(seed_text), graph


def parse_log(path):
    dataset, seed, graph = parse_log_name(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = HORIZON_RE.search(line)
        if match:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "graph": graph,
                    "horizon": int(match.group("horizon")),
                    "mae": float(match.group("mae")),
                    "rmse": float(match.group("rmse")),
                    "mape": float(match.group("mape")),
                    "wmape": float(match.group("wmape")),
                }
            )
            continue
        match = AVG_RE.search(line)
        if match:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "graph": graph,
                    "horizon": "avg",
                    "mae": float(match.group("mae")),
                    "rmse": float(match.group("rmse")),
                    "mape": float(match.group("mape")),
                    "wmape": float(match.group("wmape")),
                }
            )
    return rows


def write_markdown(summary, out_md, log_dir):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Graph Sensitivity Results\n\n")
        f.write(f"Source logs: `{log_dir}`\n\n")
        rounded = summary.round(6)
        columns = list(rounded.columns)
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for _, row in rounded.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Collect graph sensitivity results from suite logs.")
    parser.add_argument("--log_dir", type=str, default="graph_sensitivity_logs")
    parser.add_argument("--out_csv", type=str, default="review/graph_sensitivity_results.csv")
    parser.add_argument("--out_md", type=str, default="review/graph_sensitivity_results.md")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        rows.extend(parse_log(path))
    if not rows:
        raise RuntimeError(f"No test metrics found in {log_dir}")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["dataset", "graph", "horizon"], as_index=False)[["mae", "rmse", "mape", "wmape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    write_markdown(summary, Path(args.out_md), log_dir)
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
