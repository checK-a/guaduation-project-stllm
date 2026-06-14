import argparse
import subprocess
import sys
from pathlib import Path


HORIZONS = [4, 8, 12]


def run_command(cmd, dry_run=False):
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def package_exists(dataset_name):
    root = Path("dataset") / dataset_name / dataset_name
    return (root / "train.npz").exists() and (root / "val.npz").exists() and (root / "test.npz").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset versions for missing-value interpolation sensitivity experiments."
    )
    parser.add_argument("--horizons", type=str, default="4,8,12", help="comma-separated horizons")
    parser.add_argument("--force", action="store_true", help="rebuild even if package already exists")
    parser.add_argument("--dry_run", action="store_true", help="print commands without running them")
    parser.add_argument(
        "--ili_csv",
        type=str,
        default="dataset/ili_us_states_h12/raw/cdc_ili_states_2013W40_2023W40.csv",
    )
    parser.add_argument(
        "--ili_adj_csv",
        type=str,
        default="dataset/ili_us_states_h12/raw/us_state_adjacency_edges.csv",
    )
    parser.add_argument(
        "--nhsn_csv",
        type=str,
        default="dataset/us_states_nhsn_flu_hosp_h12/raw/target-hospital-admissions.csv",
    )
    args = parser.parse_args()

    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    for horizon in horizons:
        legacy_ili = f"ili_us_states_h{horizon}_legacy_interpolate"
        if args.force or not package_exists(legacy_ili):
            run_command(
                [
                    sys.executable,
                    "prepare_cdc_ili.py",
                    "--ili_csv",
                    args.ili_csv,
                    "--adj_csv",
                    args.ili_adj_csv,
                    "--dataset_name",
                    legacy_ili,
                    "--output_len",
                    str(horizon),
                    "--imputation_policy",
                    "interpolate",
                ],
                dry_run=args.dry_run,
            )
        else:
            print(f"Skip existing package: {legacy_ili}")

        leakfree_ili = f"ili_us_states_h{horizon}_leakfree"
        if not package_exists(leakfree_ili):
            run_command(
                [
                    sys.executable,
                    "prepare_cdc_ili.py",
                    "--ili_csv",
                    args.ili_csv,
                    "--adj_csv",
                    args.ili_adj_csv,
                    "--dataset_name",
                    leakfree_ili,
                    "--output_len",
                    str(horizon),
                    "--imputation_policy",
                    "causal_median",
                ],
                dry_run=args.dry_run,
            )

        legacy_nhsn = f"us_states_nhsn_flu_hosp_h{horizon}_legacy_interpolate"
        if args.force or not package_exists(legacy_nhsn):
            run_command(
                [
                    sys.executable,
                    "prepare_nhsn_flu_us_states.py",
                    "--source_csv",
                    args.nhsn_csv,
                    "--dataset_name",
                    legacy_nhsn,
                    "--output_len",
                    str(horizon),
                    "--missing_policy",
                    "interpolate",
                ],
                dry_run=args.dry_run,
            )
        else:
            print(f"Skip existing package: {legacy_nhsn}")

        leakfree_nhsn = f"us_states_nhsn_flu_hosp_h{horizon}_leakfree"
        if not package_exists(leakfree_nhsn):
            run_command(
                [
                    sys.executable,
                    "prepare_nhsn_flu_us_states.py",
                    "--source_csv",
                    args.nhsn_csv,
                    "--dataset_name",
                    leakfree_nhsn,
                    "--output_len",
                    str(horizon),
                    "--missing_policy",
                    "causal_median",
                ],
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
