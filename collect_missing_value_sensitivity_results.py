import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_RE = re.compile(
    r"(?P<dataset>(?:ili_us_states_h\d+|us_states_nhsn_flu_hosp_h\d+)(?:_(?:leakfree|legacy_interpolate))?)"
)
SEED_RE = re.compile(r"seed(?P<seed>\d+)")
VARIANT_RE = re.compile(r"missing_value_sensitivity_(?P<variant>.+?)_seed\d+_")


def infer_dataset(path):
    match = DATASET_RE.search(str(path))
    return match.group("dataset") if match else "unknown"


def infer_seed(run_name):
    match = SEED_RE.search(run_name)
    return int(match.group("seed")) if match else None


def infer_variant(run_name):
    match = VARIANT_RE.search(run_name)
    if match:
        return match.group("variant")
    for variant in [
        "legacy_interpolate",
        "leakfree_point_mask",
        "leakfree_drop_sample_metric",
        "leakfree_drop_node_metric",
    ]:
        if variant in run_name:
            return variant
    return "unknown"


def read_last_row(path, prefix):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {f"{prefix}_{key}": value for key, value in row.items() if not str(key).startswith("Unnamed")}


def mask_stats(dataset_name):
    package = Path("dataset") / dataset_name / dataset_name
    rows = {}
    if not package.exists():
        return rows
    for split in ["train", "val", "test"]:
        npz_path = package / f"{split}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        for mask_name in ["x_mask", "y_mask"]:
            key = f"{split}_{mask_name}"
            if mask_name not in data:
                rows[f"{key}_missing_count"] = np.nan
                rows[f"{key}_affected_samples"] = np.nan
                continue
            mask = data[mask_name].astype(bool)
            rows[f"{key}_missing_count"] = int((~mask).sum())
            rows[f"{key}_affected_samples"] = int((~mask.reshape(mask.shape[0], -1)).any(axis=1).sum())
    meta_path = package / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        rows["imputation_policy"] = meta.get("imputation_policy", meta.get("missing_policy", ""))
        rows["missing_value_count"] = meta.get("missing_value_count", np.nan)
    return rows


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
    metric_cols = [
        "test_test_loss",
        "test_test_rmse",
        "test_test_mape",
        "test_test_wmape",
        "train_valid_loss",
        "test_y_mask_missing_count",
        "test_y_mask_affected_samples",
    ]
    present_metrics = [col for col in metric_cols if col in df.columns]
    summary = (
        df.groupby(["dataset_family", "horizon", "variant"], dropna=False)[present_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    raw_cols = [
        "dataset",
        "dataset_family",
        "horizon",
        "variant",
        "seed",
        "test_mask_policy",
        "imputation_policy",
        "test_test_loss",
        "test_test_rmse",
        "test_test_mape",
        "test_test_wmape",
        "test_y_mask_missing_count",
        "test_y_mask_affected_samples",
        "run",
    ]
    raw_cols = [col for col in raw_cols if col in df.columns]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Missing-Value Interpolation Sensitivity Results\n\n")
        f.write(
            "Variants compare legacy full-panel interpolation against leak-free causal imputation "
            "with point-wise masked targets and stricter test-mask policies.\n\n"
        )
        f.write("## Summary\n\n")
        write_table(f, summary)
        f.write("\n## Per Run\n\n")
        write_table(f, df[raw_cols])


def dataset_family(dataset):
    if dataset.startswith("ili_"):
        return "ili"
    if dataset.startswith("us_states_nhsn"):
        return "nhsn"
    return "unknown"


def horizon_from_dataset(dataset):
    match = re.search(r"_h(\d+)", dataset)
    return int(match.group(1)) if match else None


def test_mask_policy_from_run(run):
    if "drop_sample" in run:
        return "drop_sample"
    if "drop_node" in run:
        return "drop_node"
    return "point"


def main():
    parser = argparse.ArgumentParser(description="Collect missing-value sensitivity experiment results.")
    parser.add_argument("--root", type=str, default="logs", help="root directory to scan")
    parser.add_argument("--pattern", type=str, default="**/test.csv")
    parser.add_argument("--out_csv", type=str, default="review/missing_value_sensitivity_results.csv")
    parser.add_argument("--out_md", type=str, default="review/missing_value_sensitivity_results.md")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for test_path in sorted(root.glob(args.pattern)):
        run_dir = test_path.parent
        run_name = run_dir.name
        if "missing_value_sensitivity_" not in run_name:
            continue
        dataset = infer_dataset(test_path)
        row = {
            "path": str(test_path),
            "run": run_name,
            "dataset": dataset,
            "dataset_family": dataset_family(dataset),
            "horizon": horizon_from_dataset(dataset),
            "variant": infer_variant(run_name),
            "seed": infer_seed(run_name),
            "test_mask_policy": test_mask_policy_from_run(run_name),
        }
        row.update(read_last_row(run_dir / "train.csv", "train"))
        row.update(read_last_row(test_path, "test"))
        row.update(mask_stats(dataset))
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No missing-value sensitivity runs found under {root} with pattern {args.pattern}")

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_markdown(out, Path(args.out_md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
