import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_RE = re.compile(
    r"(?P<dataset>(?:ili_us_states_h\d+|us_states_nhsn_flu_hosp_h\d+)(?:_leakfree)?)"
)
SEED_RE = re.compile(r"seed(?P<seed>\d+)")
VARIANT_RE = re.compile(r"sir_mass_ablation_(?P<variant>.+?)_seed\d+_")


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
    for variant in ["lambda_mass_0p01", "lambda_mass_0", "no_mech_regularizers"]:
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
        "train_train_mass",
        "train_valid_mass",
        "mass_abs_drift_mean",
        "mass_abs_drift_max",
        "mass_rel_drift_mean",
        "infection_clip_ratio",
        "recovery_clip_ratio",
        "beta_mean",
        "gamma_mean",
    ]
    present_metrics = [col for col in metric_cols if col in df.columns]
    summary = (
        df.groupby(["dataset", "variant", "split"], dropna=False)[present_metrics]
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
        "split",
        "test_test_loss",
        "test_test_rmse",
        "test_test_mape",
        "test_test_wmape",
        "train_train_mass",
        "train_valid_mass",
        "mass_abs_drift_mean",
        "mass_abs_drift_max",
        "mass_rel_drift_mean",
        "infection_clip_ratio",
        "recovery_clip_ratio",
        "run",
    ]
    raw_cols = [col for col in raw_cols if col in df.columns]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Latent SIR Mass-Loss Ablation Results\n\n")
        f.write(
            "Diagnostics are computed from the loaded evaluation checkpoint. "
            "`mass_abs_drift_*` measures the change in latent `S+I+R` over rollout horizons.\n\n"
        )
        f.write("## Summary\n\n")
        write_table(f, summary)
        f.write("\n## Per Split\n\n")
        write_table(f, df[raw_cols])


def main():
    parser = argparse.ArgumentParser(description="Collect latent SIR mass-loss ablation diagnostics.")
    parser.add_argument("--root", type=str, default="logs", help="root directory to scan")
    parser.add_argument("--pattern", type=str, default="**/sir_diagnostics.csv")
    parser.add_argument("--out_csv", type=str, default="review/sir_mass_ablation_results.csv")
    parser.add_argument("--out_md", type=str, default="review/sir_mass_ablation_results.md")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob(args.pattern))
    rows = []
    for diag_path in paths:
        run_dir = diag_path.parent
        run_name = run_dir.name
        diag = pd.read_csv(diag_path)
        train_last = read_last_row(run_dir / "train.csv", "train")
        test_last = read_last_row(run_dir / "test.csv", "test")
        for _, diag_row in diag.iterrows():
            row = {
                "path": str(diag_path),
                "run": run_name,
                "dataset": infer_dataset(diag_path),
                "variant": infer_variant(run_name),
                "seed": infer_seed(run_name),
            }
            row.update(train_last)
            row.update(test_last)
            row.update(diag_row.to_dict())
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No SIR diagnostic CSVs found under {root} with pattern {args.pattern}")

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_markdown(out, Path(args.out_md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
