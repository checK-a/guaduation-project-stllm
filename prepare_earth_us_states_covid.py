import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EARTH US state COVID matrix files into this project's npz dataset format."
    )
    parser.add_argument(
        "--data_txt",
        type=str,
        required=True,
        help="Path to EARTH daily US state COVID matrix .txt file.",
    )
    parser.add_argument(
        "--adj_txt",
        type=str,
        required=True,
        help="Path to EARTH adjacency .txt file aligned to the same state order.",
    )
    parser.add_argument(
        "--meta_json",
        type=str,
        required=True,
        help="Path to EARTH metadata .json file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="us_states_covid_jhu_20200301_20210630_ma7",
        help="Output dataset name.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset",
        help="Root directory under which the converted dataset will be written.",
    )
    parser.add_argument("--input_len", type=int, default=24, help="Number of input days.")
    parser.add_argument("--output_len", type=int, default=4, help="Number of prediction days.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    return parser.parse_args()


def validate_args(args):
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input_len and output_len must be positive.")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")


def load_source(data_txt_path: Path, adj_txt_path: Path, meta_json_path: Path):
    matrix = np.loadtxt(data_txt_path, delimiter=",", dtype=np.float32)
    adj = np.loadtxt(adj_txt_path, delimiter=",", dtype=np.float32)
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {data_txt_path}, got shape {matrix.shape}")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected square adjacency matrix in {adj_txt_path}, got shape {adj.shape}")
    if matrix.shape[1] != adj.shape[0]:
        raise ValueError(
            f"Data matrix node count {matrix.shape[1]} does not match adjacency size {adj.shape[0]}"
        )
    return matrix, adj, meta


def build_daily_index(meta, num_steps):
    start = pd.to_datetime(meta["date_start"])
    end = pd.to_datetime(meta["date_end"])
    dates = pd.date_range(start=start, end=end, freq="D")
    if len(dates) != num_steps:
        raise ValueError(
            f"Date range {meta['date_start']} -> {meta['date_end']} yields {len(dates)} days, "
            f"but matrix has {num_steps} rows."
        )
    date_df = pd.DataFrame({"date": dates})
    date_df["date_id"] = np.arange(len(date_df), dtype=np.int32)
    date_df["date_str"] = date_df["date"].dt.strftime("%Y-%m-%d")
    # ISO week 1..53 -> zero-based 0..52 for compatibility with the current week embedding.
    date_df["week_idx"] = (date_df["date"].dt.isocalendar().week.astype(np.int32) - 1).clip(0, 52)
    return date_df


def build_processed_panel(matrix, date_df, state_order):
    records = []
    for state_id, state_name in enumerate(state_order):
        state_series = matrix[:, state_id]
        state_df = date_df.copy()
        state_df["state_id"] = state_id
        state_df["state_name"] = state_name
        state_df["cases_ma7"] = state_series.astype(np.float32)
        records.append(state_df)
    panel = pd.concat(records, ignore_index=True)
    return panel[
        ["date_id", "date_str", "week_idx", "state_id", "state_name", "cases_ma7"]
    ].copy()


def make_windows(matrix, week_indices, date_df, input_len, output_len):
    features = matrix[:, :, None].astype(np.float32)
    num_steps = features.shape[0]
    num_samples = num_steps - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({num_steps}) for input_len={input_len} and output_len={output_len}"
        )

    xs = []
    ys = []
    week_idx_xs = []
    week_idx_ys = []
    sample_ranges = []
    date_strings = date_df["date_str"].tolist()

    for start_idx in range(num_samples):
        input_end = start_idx + input_len
        target_end = input_end + output_len
        xs.append(features[start_idx:input_end])
        ys.append(features[input_end:target_end])
        week_idx_xs.append(week_indices[start_idx:input_end])
        week_idx_ys.append(week_indices[input_end:target_end])
        sample_ranges.append(
            {
                "sample_id": start_idx,
                "input_start": date_strings[start_idx],
                "input_end": date_strings[input_end - 1],
                "target_start": date_strings[input_end],
                "target_end": date_strings[target_end - 1],
            }
        )

    return (
        np.stack(xs),
        np.stack(ys),
        np.stack(week_idx_xs).astype(np.int64),
        np.stack(week_idx_ys).astype(np.int64),
        sample_ranges,
    )


def split_windows(xs, ys, week_idx_xs, week_idx_ys, sample_ranges, train_ratio, val_ratio, output_len):
    num_samples = xs.shape[0]
    gap = max(output_len - 1, 0)
    effective_samples = num_samples - 2 * gap
    if effective_samples <= 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) to apply chronological split with output_len={output_len}"
        )

    train_count = int(effective_samples * train_ratio)
    val_count = int(effective_samples * val_ratio)
    test_count = effective_samples - train_count - val_count

    train_end = train_count
    val_start = train_end + gap
    val_end = val_start + val_count
    test_start = val_end + gap

    if (
        train_count <= 0
        or val_count <= 0
        or test_count <= 0
        or val_end > num_samples
        or test_start >= num_samples
    ):
        raise ValueError(
            "Invalid chronological split sizes for "
            f"{num_samples} samples with output_len={output_len}: "
            f"train_count={train_count}, val_count={val_count}, test_count={test_count}"
        )

    split_indices = {
        "train": (0, train_end),
        "val": (val_start, val_end),
        "test": (test_start, num_samples),
    }
    split_data = {}
    for split_name, (start, end) in split_indices.items():
        split_data[split_name] = {
            "x": xs[start:end],
            "y": ys[start:end],
            "week_idx_x": week_idx_xs[start:end],
            "week_idx_y": week_idx_ys[start:end],
            "sample_ranges": sample_ranges[start:end],
        }
    return split_data


def save_npz_splits(output_dir, split_data):
    for split_name, payload in split_data.items():
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            x=payload["x"],
            y=payload["y"],
            week_idx_x=payload["week_idx_x"],
            week_idx_y=payload["week_idx_y"],
        )


def copy_raw_inputs(raw_dir: Path, source_paths):
    for src_path in source_paths:
        dst_path = raw_dir / src_path.name
        if src_path.resolve() == dst_path.resolve():
            continue
        shutil.copy2(src_path, dst_path)


def build_quality_report(matrix, date_df, state_order, adj):
    per_state = []
    for state_id, state_name in enumerate(state_order):
        series = matrix[:, state_id]
        per_state.append(
            {
                "state_name": state_name,
                "num_days": int(len(series)),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "mean_value": float(series.mean()),
            }
        )

    report = {
        "num_states": len(state_order),
        "num_days": len(date_df),
        "start_date": date_df.iloc[0]["date_str"],
        "end_date": date_df.iloc[-1]["date_str"],
        "adjacency_shape": list(adj.shape),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_diagonal_all_ones": bool(np.all(np.diag(adj) == 1.0)),
        "per_state": per_state,
    }
    return report


def main():
    args = parse_args()
    validate_args(args)

    data_txt_path = Path(args.data_txt).resolve()
    adj_txt_path = Path(args.adj_txt).resolve()
    meta_json_path = Path(args.meta_json).resolve()
    for path in [data_txt_path, adj_txt_path, meta_json_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    dataset_root = Path(args.output_root).resolve() / args.dataset_name
    raw_dir = ensure_dir(dataset_root / "raw")
    processed_dir = ensure_dir(dataset_root / "processed")
    package_dir = ensure_dir(dataset_root / args.dataset_name)

    matrix, adj, source_meta = load_source(data_txt_path, adj_txt_path, meta_json_path)
    state_order = source_meta["state_order"]
    date_df = build_daily_index(source_meta, matrix.shape[0])
    panel = build_processed_panel(matrix, date_df, state_order)
    week_indices = date_df["week_idx"].to_numpy(dtype=np.int32)

    xs, ys, week_idx_xs, week_idx_ys, sample_ranges = make_windows(
        matrix, week_indices, date_df, args.input_len, args.output_len
    )
    split_data = split_windows(
        xs,
        ys,
        week_idx_xs,
        week_idx_ys,
        sample_ranges,
        args.train_ratio,
        args.val_ratio,
        args.output_len,
    )

    copy_raw_inputs(raw_dir, [data_txt_path, adj_txt_path, meta_json_path])
    panel.to_csv(processed_dir / "panel.csv", index=False)
    pd.DataFrame(
        {"state_id": np.arange(len(state_order), dtype=np.int32), "state_name": state_order}
    ).to_csv(processed_dir / "state_index.csv", index=False)

    quality_report = build_quality_report(matrix, date_df, state_order, adj)
    with open(processed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    save_npz_splits(package_dir, split_data)
    with open(package_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f)

    train_value_channel = split_data["train"]["x"][..., 0]
    meta = {
        "dataset_name": args.dataset_name,
        "source_dataset_name": source_meta.get("dataset_name"),
        "date_start": date_df.iloc[0]["date_str"],
        "date_end": date_df.iloc[-1]["date_str"],
        "num_nodes": len(state_order),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "feature_names": ["daily_new_confirmed_ma7"],
        "state_order": state_order,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "scaler_mean": float(train_value_channel.mean()),
        "scaler_std": float(train_value_channel.std()),
        "time_index_type": "daily",
        "week_index_semantics": "iso_week_zero_based_0_52",
        "source_meta": source_meta,
        "splits": {
            split_name: {
                "num_samples": int(payload["x"].shape[0]),
                "first_sample": payload["sample_ranges"][0],
                "last_sample": payload["sample_ranges"][-1],
            }
            for split_name, payload in split_data.items()
        },
    }
    with open(package_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Prepared dataset written to: {dataset_root}")
    print(
        "Package shapes:",
        {name: {"x": payload["x"].shape, "y": payload["y"].shape} for name, payload in split_data.items()},
    )
    print(f"Adjacency shape: {adj.shape}")
    print(f"Date range: {date_df.iloc[0]['date_str']} -> {date_df.iloc[-1]['date_str']}")


if __name__ == "__main__":
    main()
