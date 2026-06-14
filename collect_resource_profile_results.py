import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_RE = re.compile(
    r"(?P<dataset>(?:ili_us_states_h\d+|us_states_nhsn_flu_hosp_h\d+)(?:_leakfree)?)"
)
VARIANT_RE = re.compile(r"resource_profile_(?P<variant>.+?)_seed\d+_")


def infer_dataset_from_path(path):
    match = DATASET_RE.search(str(path))
    return match.group("dataset") if match else "unknown"


def infer_variant_from_run(run_name):
    match = VARIANT_RE.search(run_name)
    if match:
        return match.group("variant")
    for variant in ["full", "without_llm", "vanilla_transformer", "random_init_gpt2"]:
        if variant in run_name:
            return variant
    return "unknown"


def write_table(f, df):
    if df.empty:
        f.write("_No rows._\n")
        return
    rounded = df.round(6)
    columns = list(rounded.columns)
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
    for _, row in rounded.iterrows():
        f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")


def write_markdown(df, out_md):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    group_cols = ["dataset", "variant", "model", "output_len"]
    metric_cols = [
        "epochs_completed",
        "total_params",
        "trainable_params",
        "avg_train_sec_per_epoch",
        "avg_val_sec_per_epoch",
        "test_eval_sec",
        "train_samples_per_sec",
        "test_samples_per_sec",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
    ]
    present_metrics = [col for col in metric_cols if col in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)[present_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    raw_cols = [
        "dataset",
        "variant",
        "seed",
        "model",
        "output_len",
        "epochs_completed",
        "best_epoch",
        "total_params",
        "trainable_params",
        "avg_train_sec_per_epoch",
        "avg_val_sec_per_epoch",
        "test_eval_sec",
        "train_samples_per_sec",
        "test_samples_per_sec",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
        "run_dir",
    ]
    raw_cols = [col for col in raw_cols if col in df.columns]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Resource Profile Results\n\n")
        f.write(
            "Each row is produced by `train_plus.py --profile_resources true`. "
            "CUDA timings are synchronized before/after train, validation, and test sections.\n\n"
        )
        f.write("## Summary\n\n")
        write_table(f, summary)
        f.write("\n## Per Run\n\n")
        write_table(f, df[raw_cols])


def main():
    parser = argparse.ArgumentParser(description="Collect resource profile CSVs from training logs.")
    parser.add_argument("--root", type=str, default="logs", help="root directory to scan")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/resource_report.csv",
        help="glob pattern under root",
    )
    parser.add_argument("--out_csv", type=str, default="review/resource_profile_results.csv")
    parser.add_argument("--out_md", type=str, default="review/resource_profile_results.md")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob(args.pattern))
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        run_name = path.parent.name
        df.insert(0, "run", run_name)
        df.insert(0, "variant", infer_variant_from_run(run_name))
        df.insert(0, "dataset_from_path", infer_dataset_from_path(path))
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No resource profile CSVs found under {root} with pattern {args.pattern}")

    out = pd.concat(rows, ignore_index=True)
    if "dataset" not in out.columns:
        out["dataset"] = out["dataset_from_path"]
    else:
        out["dataset"] = out["dataset"].fillna(out["dataset_from_path"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_markdown(out, Path(args.out_md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
