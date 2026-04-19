import argparse
import csv
import io
import json
import pickle
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


US_STATE_NAMES = {
    "ak": "Alaska",
    "al": "Alabama",
    "ar": "Arkansas",
    "az": "Arizona",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "dc": "District of Columbia",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "ia": "Iowa",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "ma": "Massachusetts",
    "md": "Maryland",
    "me": "Maine",
    "mi": "Michigan",
    "mn": "Minnesota",
    "mo": "Missouri",
    "ms": "Mississippi",
    "mt": "Montana",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "ne": "Nebraska",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "nv": "Nevada",
    "ny": "New York",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "va": "Virginia",
    "vt": "Vermont",
    "wa": "Washington",
    "wi": "Wisconsin",
    "wv": "West Virginia",
    "wy": "Wyoming",
}

STATE_BORDERS = {
    "al": {"fl", "ga", "ms", "tn"},
    "ak": set(),
    "ar": {"mo", "tn", "ms", "la", "tx", "ok"},
    "az": {"ca", "nv", "ut", "nm", "co"},
    "ca": {"or", "nv", "az"},
    "co": {"wy", "ne", "ks", "ok", "nm", "az", "ut"},
    "ct": {"ny", "ma", "ri"},
    "dc": {"md", "va"},
    "de": {"md", "nj", "pa"},
    "fl": {"al", "ga"},
    "ga": {"fl", "al", "tn", "nc", "sc"},
    "hi": set(),
    "ia": {"mn", "sd", "ne", "mo", "il", "wi"},
    "id": {"wa", "or", "nv", "ut", "wy", "mt"},
    "il": {"wi", "ia", "mo", "ky", "in"},
    "in": {"mi", "oh", "ky", "il"},
    "ks": {"ne", "mo", "ok", "co"},
    "ky": {"il", "in", "oh", "wv", "va", "tn", "mo"},
    "la": {"tx", "ar", "ms"},
    "ma": {"ri", "ct", "ny", "vt", "nh"},
    "md": {"va", "wv", "pa", "de", "dc"},
    "me": {"nh"},
    "mi": {"wi", "in", "oh"},
    "mn": {"nd", "sd", "ia", "wi"},
    "mo": {"ia", "il", "ky", "tn", "ar", "ok", "ks", "ne"},
    "ms": {"la", "ar", "tn", "al"},
    "mt": {"id", "wy", "sd", "nd"},
    "nc": {"va", "tn", "ga", "sc"},
    "nd": {"mt", "sd", "mn"},
    "ne": {"sd", "ia", "mo", "ks", "co", "wy"},
    "nh": {"me", "ma", "vt"},
    "nj": {"ny", "de", "pa"},
    "nm": {"az", "ut", "co", "ok", "tx"},
    "nv": {"or", "id", "ut", "az", "ca"},
    "ny": {"pa", "nj", "ct", "ma", "vt"},
    "oh": {"mi", "pa", "wv", "ky", "in"},
    "ok": {"co", "ks", "mo", "ar", "tx", "nm"},
    "or": {"wa", "id", "nv", "ca"},
    "pa": {"ny", "nj", "de", "md", "wv", "oh"},
    "ri": {"ct", "ma"},
    "sc": {"nc", "ga"},
    "sd": {"nd", "mn", "ia", "ne", "wy", "mt"},
    "tn": {"ky", "va", "nc", "ga", "al", "ms", "ar", "mo"},
    "tx": {"nm", "ok", "ar", "la"},
    "ut": {"id", "wy", "co", "nm", "az", "nv"},
    "va": {"nc", "tn", "ky", "wv", "md", "dc"},
    "vt": {"ny", "nh", "ma"},
    "wa": {"id", "or"},
    "wi": {"mn", "ia", "il", "mi"},
    "wv": {"oh", "pa", "md", "va", "ky"},
    "wy": {"mt", "sd", "ne", "co", "ut", "id"},
}

EXCLUDED_ENTITIES = {
    "American Samoa",
    "Diamond Princess",
    "Grand Princess",
    "Guam",
    "Northern Mariana Islands",
    "Puerto Rico",
    "Recovered",
    "Virgin Islands",
}

ORDERED_STATE_CODES = [code for code, _ in sorted(US_STATE_NAMES.items(), key=lambda item: item[1])]


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_jhu_date(value: str) -> date:
    return datetime.strptime(value, "%m/%d/%y").date()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare a JHU US states COVID dataset in this project's npz format."
    )
    parser.add_argument("--source_csv", type=str, required=True, help="Path to JHU US confirmed cases CSV.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="us_states_covid_jhu_20200301_20230309_ma7",
        help="Output dataset name.",
    )
    parser.add_argument("--output_root", type=str, default="dataset", help="Output root directory.")
    parser.add_argument("--start_date", type=str, default="2020-03-01", help="Inclusive start date.")
    parser.add_argument("--end_date", type=str, default="2023-03-09", help="Inclusive end date.")
    parser.add_argument("--smoothing_window", type=int, default=7, help="Trailing moving-average window.")
    parser.add_argument(
        "--negative_policy",
        type=str,
        default="clip",
        choices=["clip", "retain", "error"],
        help="How to handle negative daily diffs caused by source revisions.",
    )
    parser.add_argument("--include_dc", type=str, choices=["true", "false"], default="true")
    parser.add_argument("--input_len", type=int, default=24, help="Number of input days.")
    parser.add_argument("--output_len", type=int, default=4, help="Number of prediction days.")
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


def trailing_moving_average(matrix: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.zeros_like(matrix, dtype=np.float32)
    for idx in range(matrix.shape[0]):
        start = max(0, idx - window + 1)
        smoothed[idx] = matrix[start : idx + 1].mean(axis=0, dtype=np.float32)
    return smoothed


def cumulative_to_daily(cumulative: np.ndarray, negative_policy: str) -> tuple[np.ndarray, int]:
    daily = np.empty_like(cumulative, dtype=np.float32)
    daily[0] = cumulative[0]
    if cumulative.shape[0] > 1:
        daily[1:] = cumulative[1:] - cumulative[:-1]
    negative_count = int(np.sum(daily < 0))
    if negative_count and negative_policy == "error":
        raise RuntimeError("Negative daily diffs detected in the JHU source.")
    if negative_policy == "clip":
        daily = np.maximum(daily, 0.0)
    return daily.astype(np.float32, copy=False), negative_count


def parse_jhu_rows(csv_text: str, include_dc: bool):
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise RuntimeError("The source CSV is empty or missing headers.")

    date_columns = reader.fieldnames[11:]
    parsed_dates = [parse_jhu_date(column) for column in date_columns]

    expected_codes = [code for code in ORDERED_STATE_CODES if include_dc or code != "dc"]
    expected_names = {US_STATE_NAMES[code] for code in expected_codes}
    cumulative_by_state = {
        state_name: np.zeros(len(parsed_dates), dtype=np.float32) for state_name in expected_names
    }
    county_counts = {state_name: 0 for state_name in expected_names}
    seen_non_state = set()

    for row in reader:
        state_name = (row.get("Province_State") or "").strip()
        if state_name in EXCLUDED_ENTITIES:
            continue
        if state_name not in expected_names:
            seen_non_state.add(state_name)
            continue
        values = np.array([float(row[column] or 0.0) for column in date_columns], dtype=np.float32)
        cumulative_by_state[state_name] += values
        county_counts[state_name] += 1

    missing_states = sorted(name for name, count in county_counts.items() if count == 0)
    if missing_states:
        raise RuntimeError(f"The source CSV is missing expected states: {missing_states}")

    return parsed_dates, cumulative_by_state, county_counts, seen_non_state


def build_state_matrix(
    parsed_dates,
    cumulative_by_state,
    include_dc,
    start_date,
    end_date,
    negative_policy,
    smoothing_window,
):
    date_to_index = {value: idx for idx, value in enumerate(parsed_dates)}
    if start_date not in date_to_index:
        raise RuntimeError(f"start_date {start_date.isoformat()} is not available in the JHU CSV.")
    if end_date not in date_to_index:
        raise RuntimeError(f"end_date {end_date.isoformat()} is not available in the JHU CSV.")

    start_idx = date_to_index[start_date]
    end_idx = date_to_index[end_date]
    state_codes = [code for code in ORDERED_STATE_CODES if include_dc or code != "dc"]
    state_names = [US_STATE_NAMES[code] for code in state_codes]

    matrix = np.zeros((end_idx - start_idx + 1, len(state_codes)), dtype=np.float32)
    raw_rows = []
    negative_counts = {}

    for column, (state_code, state_name) in enumerate(zip(state_codes, state_names)):
        cumulative = cumulative_by_state[state_name]
        daily, negative_count = cumulative_to_daily(cumulative, negative_policy)
        smoothed = trailing_moving_average(daily.reshape(-1, 1), smoothing_window).reshape(-1)
        negative_counts[state_code] = negative_count
        matrix[:, column] = smoothed[start_idx : end_idx + 1]

        for offset, current_date in enumerate(parsed_dates[start_idx : end_idx + 1]):
            source_idx = start_idx + offset
            raw_rows.append(
                {
                    "date": current_date.isoformat(),
                    "region": state_code,
                    "region_name": state_name,
                    "cumulative_confirmed": float(cumulative[source_idx]),
                    "daily_new_confirmed": float(daily[source_idx]),
                    f"daily_new_confirmed_ma{smoothing_window}": float(smoothed[source_idx]),
                }
            )

    return parsed_dates[start_idx : end_idx + 1], matrix, raw_rows, negative_counts, state_codes, state_names


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


def build_processed_panel(matrix, date_df, state_order):
    records = []
    for state_id, state_name in enumerate(state_order):
        state_df = date_df.copy()
        state_df["state_id"] = state_id
        state_df["state_name"] = state_name
        state_df["daily_new_confirmed_ma7"] = matrix[:, state_id].astype(np.float32)
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
            "daily_new_confirmed_ma7",
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


def build_quality_report(matrix, date_df, state_order, adj, negative_counts):
    per_state = []
    code_by_name = {name: code for code, name in US_STATE_NAMES.items()}
    for state_id, state_name in enumerate(state_order):
        series = matrix[:, state_id]
        per_state.append(
            {
                "state_name": state_name,
                "state_code": code_by_name[state_name],
                "num_days": int(len(series)),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "mean_value": float(series.mean()),
                "negative_daily_diffs": int(negative_counts.get(code_by_name[state_name], 0)),
            }
        )

    return {
        "num_states": len(state_order),
        "num_days": len(date_df),
        "start_date": date_df.iloc[0]["date_str"],
        "end_date": date_df.iloc[-1]["date_str"],
        "adjacency_shape": list(adj.shape),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_diagonal_all_ones": bool(np.all(np.diag(adj) == 1.0)),
        "per_state": per_state,
    }


def copy_raw_inputs(raw_dir: Path, source_paths):
    for src_path in source_paths:
        dst_path = raw_dir / src_path.name
        if src_path.resolve() == dst_path.resolve():
            continue
        shutil.copy2(src_path, dst_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    include_dc = args.include_dc.lower() == "true"
    source_csv_path = Path(args.source_csv).resolve()
    if not source_csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv_path}")

    csv_text = source_csv_path.read_text(encoding="utf-8")
    parsed_dates, cumulative_by_state, county_counts, ignored_entities = parse_jhu_rows(
        csv_text, include_dc
    )
    filtered_dates, matrix, raw_rows, negative_counts, state_codes, state_names = build_state_matrix(
        parsed_dates,
        cumulative_by_state,
        include_dc,
        parse_iso_date(args.start_date),
        parse_iso_date(args.end_date),
        args.negative_policy,
        args.smoothing_window,
    )

    adj = build_adjacency(state_codes)
    date_df = build_daily_index(filtered_dates)
    panel = build_processed_panel(matrix, date_df, state_names)
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
    raw_csv_path = processed_dir / "raw_daily_panel.csv"
    pd.DataFrame(raw_rows).to_csv(raw_csv_path, index=False)
    panel.to_csv(processed_dir / "panel.csv", index=False)
    pd.DataFrame(
        {"state_id": np.arange(len(state_names), dtype=np.int32), "state_name": state_names, "state_code": state_codes}
    ).to_csv(processed_dir / "state_index.csv", index=False)

    quality_report = build_quality_report(matrix, date_df, state_names, adj, negative_counts)
    with open(processed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    save_npz_splits(package_dir, split_data)
    with open(package_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f)

    train_value_channel = split_data["train"]["x"][..., 0]
    meta = {
        "dataset_name": args.dataset_name,
        "source": "JHU-CSSE COVID-19 US time series",
        "source_csv": str(source_csv_path),
        "derived_from": "time_series_covid19_confirmed_US.csv cumulative counts",
        "metric": f"daily_new_confirmed_ma{args.smoothing_window}",
        "transform": "daily_diff",
        "negative_policy": "clip_zero" if args.negative_policy == "clip" else args.negative_policy,
        "negative_daily_diffs": negative_counts,
        "smoothing": {"method": "trailing_moving_average", "window": args.smoothing_window},
        "include_dc": include_dc,
        "date_start": date_df.iloc[0]["date_str"],
        "date_end": date_df.iloc[-1]["date_str"],
        "raw_date_start": filtered_dates[0].isoformat(),
        "raw_date_end": filtered_dates[-1].isoformat(),
        "num_nodes": len(state_names),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "feature_names": [f"daily_new_confirmed_ma{args.smoothing_window}"],
        "regions": state_codes,
        "state_order": state_names,
        "county_rows_aggregated": county_counts,
        "ignored_non_state_entities": sorted(entity for entity in ignored_entities if entity),
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
