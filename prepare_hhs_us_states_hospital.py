import argparse
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_jhu_us_states_covid import ORDERED_STATE_CODES, STATE_BORDERS, US_STATE_NAMES


ABBR_TO_STATE = {code.upper(): name for code, name in US_STATE_NAMES.items()}
STATE_TO_ABBR = {name: code for code, name in US_STATE_NAMES.items()}
EXCLUDED_ENTITIES = {
    "American Samoa",
    "Guam",
    "Northern Mariana Islands",
    "Puerto Rico",
    "Virgin Islands",
    "United States",
    "National",
}

DATE_COLUMN_CANDIDATES = [
    "date",
    "collection_date",
    "reporting_date",
    "reporting_for_date",
]
STATE_COLUMN_CANDIDATES = [
    "state",
    "state_name",
    "jurisdiction",
]
ADULT_ADMISSIONS_CANDIDATES = [
    "previous_day_admission_adult_covid_confirmed",
    "previous_day_admission_adult_confirmed_covid",
]
PEDIATRIC_ADMISSIONS_CANDIDATES = [
    "previous_day_admission_pediatric_covid_confirmed",
    "previous_day_admission_pediatric_confirmed_covid",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare a state-level daily HHS COVID admissions dataset in this project's npz format."
    )
    parser.add_argument("--source_csv", type=str, required=True, help="Path to the fetched HHS CSV.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="us_states_hhs_covid_admissions_ma7_h10",
        help="Output dataset name.",
    )
    parser.add_argument("--output_root", type=str, default="dataset", help="Output root directory.")
    parser.add_argument("--start_date", type=str, default=None, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end_date", type=str, default=None, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--smoothing_window", type=int, default=7, help="Trailing moving-average window.")
    parser.add_argument(
        "--negative_policy",
        type=str,
        default="clip",
        choices=["clip", "retain", "error"],
        help="How to handle negative admissions caused by source revisions.",
    )
    parser.add_argument("--include_dc", type=parse_bool, default=True, help="Whether to include DC.")
    parser.add_argument("--input_len", type=int, default=24, help="Number of input days.")
    parser.add_argument("--output_len", type=int, default=10, help="Number of prediction days.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    return parser


def validate_args(args):
    if args.smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input_len and output_len must be positive.")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")


def normalize_header(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def detect_column(df, candidates):
    normalized_to_original = {normalize_header(col): col for col in df.columns}
    for candidate in candidates:
        normalized_candidate = normalize_header(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]
    return None


def normalize_state(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in ABBR_TO_STATE:
        return ABBR_TO_STATE[upper]
    title = re.sub(r"\s+", " ", text).title()
    if title in STATE_TO_ABBR and title not in EXCLUDED_ENTITIES:
        return title
    return None


def trailing_moving_average(matrix: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.zeros_like(matrix, dtype=np.float32)
    for idx in range(matrix.shape[0]):
        start = max(0, idx - window + 1)
        smoothed[idx] = matrix[start : idx + 1].mean(axis=0, dtype=np.float32)
    return smoothed


def build_state_order(include_dc: bool):
    return [code for code in ORDERED_STATE_CODES if include_dc or code != "dc"]


def build_adjacency(region_codes):
    adjacency = np.zeros((len(region_codes), len(region_codes)), dtype=np.float32)
    for i, region in enumerate(region_codes):
        adjacency[i, i] = 1.0
        neighbors = STATE_BORDERS.get(region, set())
        for j, other in enumerate(region_codes):
            if i == j:
                continue
            if other in neighbors or region in STATE_BORDERS.get(other, set()):
                adjacency[i, j] = 1.0
    return adjacency


def build_daily_index(filtered_dates):
    date_df = pd.DataFrame({"date": pd.to_datetime(filtered_dates)})
    date_df["date_id"] = np.arange(len(date_df), dtype=np.int32)
    date_df["date_str"] = date_df["date"].dt.strftime("%Y-%m-%d")
    date_df["week_idx"] = (date_df["date"].dt.isocalendar().week.astype(np.int32) - 1).clip(0, 52)
    date_df["dow_idx"] = date_df["date"].dt.dayofweek.astype(np.int32)
    date_df["doy_idx"] = (date_df["date"].dt.dayofyear.astype(np.int32) - 1).clip(0, 365)
    return date_df


def build_processed_panel(matrix, date_df, state_names, metric_name):
    records = []
    for state_id, state_name in enumerate(state_names):
        state_df = date_df.copy()
        state_df["state_id"] = state_id
        state_df["state_name"] = state_name
        state_df[metric_name] = matrix[:, state_id].astype(np.float32)
        records.append(state_df)
    panel = pd.concat(records, ignore_index=True)
    return panel[
        [
            "date_id",
            "date_str",
            "week_idx",
            "dow_idx",
            "doy_idx",
            "state_id",
            "state_name",
            metric_name,
        ]
    ].copy()


def make_windows(matrix, temporal_indices, date_df, input_len, output_len):
    features = matrix[:, :, None].astype(np.float32)
    num_steps = features.shape[0]
    num_samples = num_steps - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({num_steps}) for input_len={input_len} and output_len={output_len}"
        )

    xs = []
    ys = []
    temporal_idx_xs = []
    temporal_idx_ys = []
    sample_ranges = []
    date_strings = date_df["date_str"].tolist()

    for start_idx in range(num_samples):
        input_end = start_idx + input_len
        target_end = input_end + output_len
        xs.append(features[start_idx:input_end])
        ys.append(features[input_end:target_end])
        temporal_idx_xs.append(temporal_indices[start_idx:input_end])
        temporal_idx_ys.append(temporal_indices[input_end:target_end])
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
        np.stack(temporal_idx_xs).astype(np.int64),
        np.stack(temporal_idx_ys).astype(np.int64),
        sample_ranges,
    )


def split_windows(xs, ys, temporal_idx_xs, temporal_idx_ys, sample_ranges, train_ratio, val_ratio, output_len):
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
            "temporal_idx_x": temporal_idx_xs[start:end],
            "temporal_idx_y": temporal_idx_ys[start:end],
            "sample_ranges": sample_ranges[start:end],
        }
    return split_data


def save_npz_splits(output_dir, split_data):
    for split_name, payload in split_data.items():
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            x=payload["x"],
            y=payload["y"],
            temporal_idx_x=payload["temporal_idx_x"],
            temporal_idx_y=payload["temporal_idx_y"],
        )


def copy_raw_inputs(raw_dir: Path, source_paths):
    for src_path in source_paths:
        dst_path = raw_dir / src_path.name
        if src_path.resolve() == dst_path.resolve():
            continue
        shutil.copy2(src_path, dst_path)


def load_manifest_if_present(source_csv_path: Path):
    manifest_path = source_csv_path.parent / "fetch_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_clean_source(source_csv_path: Path, include_dc: bool):
    source_df = pd.read_csv(source_csv_path)

    date_col = detect_column(source_df, DATE_COLUMN_CANDIDATES)
    state_col = detect_column(source_df, STATE_COLUMN_CANDIDATES)
    adult_col = detect_column(source_df, ADULT_ADMISSIONS_CANDIDATES)
    pediatric_col = detect_column(source_df, PEDIATRIC_ADMISSIONS_CANDIDATES)

    missing = [
        name
        for name, col in {
            "date": date_col,
            "state": state_col,
            "previous_day_admission_adult_covid_confirmed": adult_col,
            "previous_day_admission_pediatric_covid_confirmed": pediatric_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"Missing required HHS columns: {missing}")

    df = source_df.rename(
        columns={
            date_col: "date",
            state_col: "state",
            adult_col: "adult_admissions",
            pediatric_col: "pediatric_admissions",
        }
    ).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["state_name"] = df["state"].map(normalize_state)
    state_order = [US_STATE_NAMES[code] for code in build_state_order(include_dc)]
    df = df.loc[df["state_name"].isin(state_order)].copy()
    if df.empty:
        raise ValueError("No state-level rows remained after filtering the HHS source.")

    df["adult_admissions"] = pd.to_numeric(df["adult_admissions"], errors="coerce")
    df["pediatric_admissions"] = pd.to_numeric(df["pediatric_admissions"], errors="coerce")
    df = df.dropna(subset=["date", "adult_admissions", "pediatric_admissions", "state_name"])
    df["daily_hospital_admissions"] = df["adult_admissions"] + df["pediatric_admissions"]

    grouped = (
        df.groupby(["date", "state_name"], as_index=False)[
            ["adult_admissions", "pediatric_admissions", "daily_hospital_admissions"]
        ]
        .sum()
        .sort_values(["date", "state_name"])
        .reset_index(drop=True)
    )
    return grouped, state_order


def determine_complete_span(pivot_df: pd.DataFrame):
    complete_rows = pivot_df.notna().all(axis=1)
    if not complete_rows.any():
        raise RuntimeError("The HHS source does not contain any date where all requested states are present.")
    first_complete = pivot_df.index[np.argmax(complete_rows.to_numpy())]
    last_complete = pivot_df.index[len(complete_rows) - 1 - np.argmax(complete_rows.to_numpy()[::-1])]
    return first_complete, last_complete


def apply_negative_policy(matrix: np.ndarray, negative_policy: str):
    negative_count = int(np.sum(matrix < 0))
    if negative_count and negative_policy == "error":
        raise RuntimeError("Negative daily admissions detected in the HHS source.")
    if negative_policy == "clip":
        matrix = np.maximum(matrix, 0.0)
    return matrix.astype(np.float32, copy=False), negative_count


def build_state_matrix(grouped_df, state_order, include_dc, start_date, end_date, negative_policy, smoothing_window):
    state_codes = build_state_order(include_dc)
    state_names = [US_STATE_NAMES[code] for code in state_codes]
    pivot = (
        grouped_df.pivot(index="date", columns="state_name", values="daily_hospital_admissions")
        .reindex(columns=state_names)
        .sort_index()
    )

    full_start, full_end = determine_complete_span(pivot)
    if start_date is None:
        start_dt = full_start
    else:
        start_dt = pd.Timestamp(start_date)
    if end_date is None:
        end_dt = full_end
    else:
        end_dt = pd.Timestamp(end_date)

    requested = pivot.loc[(pivot.index >= start_dt) & (pivot.index <= end_dt)].copy()
    if requested.empty:
        raise RuntimeError("No HHS rows remained after applying the requested date range.")

    requested_complete = requested.notna().all(axis=1)
    if not requested_complete.any():
        raise RuntimeError("The requested date range does not contain any fully observed 51-state span.")
    clipped_start = requested.index[np.argmax(requested_complete.to_numpy())]
    clipped_end = requested.index[len(requested_complete) - 1 - np.argmax(requested_complete.to_numpy()[::-1])]
    requested = requested.loc[(requested.index >= clipped_start) & (requested.index <= clipped_end)].copy()

    interior_incomplete_dates = requested.index[~requested.notna().all(axis=1)].tolist()
    if interior_incomplete_dates:
        bad_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in interior_incomplete_dates]
        raise RuntimeError(f"Incomplete state-day rows remain inside the selected span: {bad_dates[:10]}")

    raw_matrix = requested.to_numpy(dtype=np.float32)
    raw_matrix, negative_count = apply_negative_policy(raw_matrix, negative_policy)
    smoothed_matrix = trailing_moving_average(raw_matrix, smoothing_window)

    raw_rows = []
    for date_idx, current_date in enumerate(requested.index):
        for state_idx, (state_code, state_name) in enumerate(zip(state_codes, state_names)):
            raw_rows.append(
                {
                    "date": pd.Timestamp(current_date).strftime("%Y-%m-%d"),
                    "region": state_code,
                    "region_name": state_name,
                    "daily_hospital_admissions": float(raw_matrix[date_idx, state_idx]),
                    f"daily_hospital_admissions_ma{smoothing_window}": float(smoothed_matrix[date_idx, state_idx]),
                }
            )

    span_info = {
        "requested_start": None if start_date is None else str(pd.Timestamp(start_date).date()),
        "requested_end": None if end_date is None else str(pd.Timestamp(end_date).date()),
        "available_full_start": pd.Timestamp(full_start).strftime("%Y-%m-%d"),
        "available_full_end": pd.Timestamp(full_end).strftime("%Y-%m-%d"),
        "selected_start": pd.Timestamp(requested.index[0]).strftime("%Y-%m-%d"),
        "selected_end": pd.Timestamp(requested.index[-1]).strftime("%Y-%m-%d"),
        "boundary_clipped": bool(
            pd.Timestamp(requested.index[0]) != pd.Timestamp(start_dt)
            or pd.Timestamp(requested.index[-1]) != pd.Timestamp(end_dt)
        ),
        "negative_daily_admissions_total": negative_count,
    }
    return requested.index.tolist(), smoothed_matrix, raw_rows, state_codes, state_names, span_info


def build_quality_report(matrix, date_df, state_names, adj):
    per_state = []
    code_by_name = {name: code for code, name in US_STATE_NAMES.items()}
    for state_id, state_name in enumerate(state_names):
        series = matrix[:, state_id]
        per_state.append(
            {
                "state_name": state_name,
                "state_code": code_by_name[state_name],
                "num_days": int(len(series)),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "mean_value": float(series.mean()),
            }
        )

    return {
        "num_states": len(state_names),
        "num_days": len(date_df),
        "start_date": date_df.iloc[0]["date_str"],
        "end_date": date_df.iloc[-1]["date_str"],
        "adjacency_shape": list(adj.shape),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_diagonal_all_ones": bool(np.all(np.diag(adj) == 1.0)),
        "per_state": per_state,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    source_csv_path = Path(args.source_csv).resolve()
    if not source_csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv_path}")

    manifest = load_manifest_if_present(source_csv_path)
    grouped_df, state_order = load_and_clean_source(source_csv_path, args.include_dc)
    filtered_dates, matrix, raw_rows, state_codes, state_names, span_info = build_state_matrix(
        grouped_df,
        state_order,
        args.include_dc,
        args.start_date,
        args.end_date,
        args.negative_policy,
        args.smoothing_window,
    )

    adj = build_adjacency(state_codes)
    date_df = build_daily_index(filtered_dates)
    metric_name = f"daily_hospital_admissions_ma{args.smoothing_window}"
    panel = build_processed_panel(matrix, date_df, state_names, metric_name)
    temporal_indices = date_df[["dow_idx", "doy_idx"]].to_numpy(dtype=np.int32)
    xs, ys, temporal_idx_xs, temporal_idx_ys, sample_ranges = make_windows(
        matrix, temporal_indices, date_df, args.input_len, args.output_len
    )
    split_data = split_windows(
        xs,
        ys,
        temporal_idx_xs,
        temporal_idx_ys,
        sample_ranges,
        args.train_ratio,
        args.val_ratio,
        args.output_len,
    )

    dataset_root = Path(args.output_root).resolve() / args.dataset_name
    raw_dir = ensure_dir(dataset_root / "raw")
    processed_dir = ensure_dir(dataset_root / "processed")
    package_dir = ensure_dir(dataset_root / args.dataset_name)

    copy_raw_inputs(raw_dir, [source_csv_path])
    pd.DataFrame(raw_rows).to_csv(processed_dir / "raw_daily_panel.csv", index=False)
    panel.to_csv(processed_dir / "panel.csv", index=False)
    pd.DataFrame(
        {"state_id": np.arange(len(state_names), dtype=np.int32), "state_name": state_names, "state_code": state_codes}
    ).to_csv(processed_dir / "state_index.csv", index=False)

    quality_report = build_quality_report(matrix, date_df, state_names, adj)
    with open(processed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    save_npz_splits(package_dir, split_data)
    with open(package_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f)

    train_value_channel = split_data["train"]["x"][..., 0]
    meta = {
        "dataset_name": args.dataset_name,
        "source": "HHS/HealthData.gov COVID-19 Reported Patient Impact and Hospital Capacity by State Timeseries",
        "source_csv": str(source_csv_path),
        "source_manifest": manifest,
        "metric": metric_name,
        "derived_from": (
            "previous_day_admission_adult_covid_confirmed + "
            "previous_day_admission_pediatric_covid_confirmed"
        ),
        "negative_policy": "clip_zero" if args.negative_policy == "clip" else args.negative_policy,
        "smoothing": {"method": "trailing_moving_average", "window": args.smoothing_window},
        "include_dc": args.include_dc,
        "date_start": date_df.iloc[0]["date_str"],
        "date_end": date_df.iloc[-1]["date_str"],
        "full_span": span_info,
        "num_nodes": len(state_names),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "feature_names": [metric_name],
        "regions": state_codes,
        "state_order": state_names,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "scaler_mean": float(train_value_channel.mean()),
        "scaler_std": float(train_value_channel.std()),
        "time_index_type": "daily",
        "week_index_semantics": "iso_week_zero_based_0_52",
        "temporal_feature_names": ["day_of_week", "day_of_year"],
        "adjacency": {"type": "state_border", "self_loops": True, "shape": [int(adj.shape[0]), int(adj.shape[1])]},
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
