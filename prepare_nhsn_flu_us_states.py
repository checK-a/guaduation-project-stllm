import argparse
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_jhu_us_states_covid import ORDERED_STATE_CODES, STATE_BORDERS, US_STATE_NAMES


DATE_COLUMN_CANDIDATES = ["weekendingdate", "week_ending_date", "date"]
JURISDICTION_COLUMN_CANDIDATES = ["jurisdiction", "state", "location", "location_name"]
FLU_ADMISSIONS_CANDIDATES = [
    "totalconfflunewadm",
    "total_confirmed_flu_new_admissions",
    "weekly_confirmed_influenza_hospital_admissions",
    "value",
]
EXCLUDED_JURISDICTIONS = {"AS", "GU", "MP", "PR", "VI", "US", "USA"}
FIPS_TO_STATE_CODE = {
    "01": "al",
    "02": "ak",
    "04": "az",
    "05": "ar",
    "06": "ca",
    "08": "co",
    "09": "ct",
    "10": "de",
    "11": "dc",
    "12": "fl",
    "13": "ga",
    "15": "hi",
    "16": "id",
    "17": "il",
    "18": "in",
    "19": "ia",
    "20": "ks",
    "21": "ky",
    "22": "la",
    "23": "me",
    "24": "md",
    "25": "ma",
    "26": "mi",
    "27": "mn",
    "28": "ms",
    "29": "mo",
    "30": "mt",
    "31": "ne",
    "32": "nv",
    "33": "nh",
    "34": "nj",
    "35": "nm",
    "36": "ny",
    "37": "nc",
    "38": "nd",
    "39": "oh",
    "40": "ok",
    "41": "or",
    "42": "pa",
    "44": "ri",
    "45": "sc",
    "46": "sd",
    "47": "tn",
    "48": "tx",
    "49": "ut",
    "50": "vt",
    "51": "va",
    "53": "wa",
    "54": "wv",
    "55": "wi",
    "56": "wy",
}
STATE_NAME_TO_CODE = {name.lower(): code for code, name in US_STATE_NAMES.items()}


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
        description="Prepare CDC NHSN weekly state-level influenza hospital admissions for this project."
    )
    parser.add_argument("--source_csv", type=str, required=True, help="Path to fetched NHSN CSV.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="us_states_nhsn_flu_hosp_h4",
        help="Output dataset name.",
    )
    parser.add_argument("--output_root", type=str, default="dataset", help="Output root directory.")
    parser.add_argument("--start_date", type=str, default=None, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end_date", type=str, default=None, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--include_dc", type=parse_bool, default=True, help="Whether to include DC.")
    parser.add_argument("--input_len", type=int, default=24, help="Number of input weeks.")
    parser.add_argument("--output_len", type=int, default=4, help="Number of prediction weeks.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--negative_policy",
        type=str,
        default="clip",
        choices=["clip", "retain", "error"],
        help="How to handle negative values if source revisions appear.",
    )
    parser.add_argument(
        "--missing_policy",
        type=str,
        default="interpolate",
        choices=["interpolate", "zero", "error", "causal_median"],
        help="How to handle missing state-week values inside the selected span.",
    )
    return parser


def validate_args(args):
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


def build_state_order(include_dc: bool):
    return [code for code in ORDERED_STATE_CODES if include_dc or code != "dc"]


def normalize_jurisdiction(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in EXCLUDED_JURISDICTIONS:
        return None
    fips = upper.zfill(2) if upper.isdigit() else upper
    if fips in FIPS_TO_STATE_CODE:
        return FIPS_TO_STATE_CODE[fips]
    lower = upper.lower()
    if lower in US_STATE_NAMES:
        return lower
    state_name = re.sub(r"\s+", " ", text).strip().lower()
    if state_name in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[state_name]
    return None


def epiweek_label(value: pd.Timestamp) -> str:
    iso = value.isocalendar()
    return f"{int(iso.year)}W{int(iso.week):02d}"


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


def load_manifest_if_present(source_csv_path: Path):
    manifest_path = source_csv_path.parent / "fetch_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_clean_source(source_csv_path: Path, include_dc: bool):
    source_df = pd.read_csv(source_csv_path)
    date_col = detect_column(source_df, DATE_COLUMN_CANDIDATES)
    jurisdiction_col = detect_column(source_df, JURISDICTION_COLUMN_CANDIDATES)
    flu_col = detect_column(source_df, FLU_ADMISSIONS_CANDIDATES)

    missing = [
        name
        for name, col in {
            "weekendingdate": date_col,
            "jurisdiction": jurisdiction_col,
            "totalconfflunewadm": flu_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"Missing required NHSN columns: {missing}")

    state_codes = build_state_order(include_dc)
    df = source_df.rename(
        columns={
            date_col: "weekendingdate",
            jurisdiction_col: "jurisdiction",
            flu_col: "flu_admissions",
        }
    ).copy()
    df["weekendingdate"] = pd.to_datetime(df["weekendingdate"], errors="coerce").dt.tz_localize(None)
    df["region"] = df["jurisdiction"].map(normalize_jurisdiction)
    df["flu_admissions"] = pd.to_numeric(df["flu_admissions"], errors="coerce")
    df = df.loc[df["region"].isin(state_codes)].dropna(
        subset=["weekendingdate", "region", "flu_admissions"]
    )
    if df.empty:
        raise ValueError("No state-level NHSN rows remained after filtering.")

    grouped = (
        df.groupby(["weekendingdate", "region"], as_index=False)["flu_admissions"]
        .sum()
        .sort_values(["weekendingdate", "region"])
        .reset_index(drop=True)
    )
    return grouped, state_codes


def determine_complete_span(pivot_df: pd.DataFrame):
    complete_rows = pivot_df.notna().all(axis=1)
    if not complete_rows.any():
        raise RuntimeError("The NHSN source does not contain any complete state-week rows.")
    first_complete = pivot_df.index[np.argmax(complete_rows.to_numpy())]
    last_complete = pivot_df.index[len(complete_rows) - 1 - np.argmax(complete_rows.to_numpy()[::-1])]
    return first_complete, last_complete


def apply_negative_policy(matrix: np.ndarray, negative_policy: str):
    negative_count = int(np.sum(matrix < 0))
    if negative_count and negative_policy == "error":
        raise RuntimeError("Negative weekly admissions detected in the NHSN source.")
    if negative_policy == "clip":
        matrix = np.maximum(matrix, 0.0)
    return matrix.astype(np.float32, copy=False), negative_count


def causal_median_fill_frame(requested: pd.DataFrame):
    values = requested.to_numpy(dtype=np.float32)
    filled = values.copy()
    observed_mask = ~np.isnan(values)
    last_values = np.full(values.shape[1], np.nan, dtype=np.float32)
    global_values = []
    fill_sources = np.full(values.shape, "observed", dtype=object)

    for t in range(values.shape[0]):
        row = values[t]
        observed = observed_mask[t]
        current_values = row[observed]
        if current_values.size:
            current_median = float(np.median(current_values))
            global_values.extend(current_values.astype(float).tolist())
        else:
            current_median = np.nan
        global_median = float(np.median(global_values)) if global_values else np.nan

        for n in range(values.shape[1]):
            if observed[n]:
                filled[t, n] = row[n]
                fill_sources[t, n] = "observed"
            elif not np.isnan(last_values[n]):
                filled[t, n] = last_values[n]
                fill_sources[t, n] = "forward_fill"
            elif not np.isnan(current_median):
                filled[t, n] = current_median
                fill_sources[t, n] = "cross_sectional_median"
            elif not np.isnan(global_median):
                filled[t, n] = global_median
                fill_sources[t, n] = "expanding_global_median"
            else:
                filled[t, n] = 0.0
                fill_sources[t, n] = "zero_fallback"

        last_values[observed] = row[observed]

    filled_df = pd.DataFrame(filled, index=requested.index, columns=requested.columns)
    return filled_df, observed_mask.astype(bool), fill_sources


def apply_missing_policy(requested: pd.DataFrame, missing_policy: str):
    missing_mask = requested.isna()
    observed_mask = (~missing_mask).to_numpy(dtype=bool)
    missing_count = int(missing_mask.sum().sum())
    missing_by_state = {state: int(count) for state, count in missing_mask.sum(axis=0).items() if int(count) > 0}
    missing_by_week = {
        pd.Timestamp(week).strftime("%Y-%m-%d"): int(count)
        for week, count in missing_mask.sum(axis=1).items()
        if int(count) > 0
    }

    if missing_count == 0:
        return requested, {
            "missing_value_count": 0,
            "missing_policy": missing_policy,
            "missing_by_state": {},
            "missing_by_week": {},
        }, observed_mask, np.full(requested.shape, "observed", dtype=object)
    if missing_policy == "error":
        bad_weeks = list(missing_by_week)[:10]
        raise RuntimeError(f"Missing state-week rows remain inside the selected span: {bad_weeks}")
    if missing_policy == "zero":
        filled = requested.fillna(0.0)
        fill_sources = np.where(observed_mask, "observed", "zero")
    elif missing_policy == "causal_median":
        filled, observed_mask, fill_sources = causal_median_fill_frame(requested)
    else:
        filled = requested.interpolate(method="linear", limit_direction="both").ffill().bfill()
        fill_sources = np.where(observed_mask, "observed", "linear_interpolate")

    if filled.isna().any().any():
        raise RuntimeError("Missing values remain after applying missing_policy.")

    return filled, {
        "missing_value_count": missing_count,
        "missing_policy": missing_policy,
        "missing_by_state": missing_by_state,
        "missing_by_week": missing_by_week,
    }, observed_mask, fill_sources


def build_week_matrix(grouped_df, state_codes, start_date, end_date, negative_policy, missing_policy):
    pivot = (
        grouped_df.pivot(index="weekendingdate", columns="region", values="flu_admissions")
        .reindex(columns=state_codes)
        .sort_index()
    )

    full_start, full_end = determine_complete_span(pivot)
    start_dt = full_start if start_date is None else pd.Timestamp(start_date)
    end_dt = full_end if end_date is None else pd.Timestamp(end_date)

    requested = pivot.loc[(pivot.index >= start_dt) & (pivot.index <= end_dt)].copy()
    if requested.empty:
        raise RuntimeError("No NHSN rows remained after applying the requested date range.")

    requested_complete = requested.notna().all(axis=1)
    if not requested_complete.any():
        raise RuntimeError("The requested date range does not contain any complete state-week span.")
    clipped_start = requested.index[np.argmax(requested_complete.to_numpy())]
    clipped_end = requested.index[len(requested_complete) - 1 - np.argmax(requested_complete.to_numpy()[::-1])]
    requested = requested.loc[(requested.index >= clipped_start) & (requested.index <= clipped_end)].copy()

    requested, missing_info, observed_mask, fill_sources = apply_missing_policy(requested, missing_policy)

    matrix = requested.to_numpy(dtype=np.float32)
    matrix, negative_count = apply_negative_policy(matrix, negative_policy)
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
        "negative_weekly_admissions_total": negative_count,
        **missing_info,
    }
    return requested.index.tolist(), matrix, observed_mask.astype(bool), fill_sources, span_info


def build_week_index(weekending_dates):
    week_df = pd.DataFrame({"weekendingdate": pd.to_datetime(weekending_dates)})
    week_df["week_id"] = np.arange(len(week_df), dtype=np.int32)
    week_df["week_end_date"] = week_df["weekendingdate"].dt.strftime("%Y-%m-%d")
    week_df["year"] = week_df["weekendingdate"].dt.isocalendar().year.astype(np.int32)
    week_df["week"] = week_df["weekendingdate"].dt.isocalendar().week.astype(np.int32)
    week_df["epiweek"] = week_df["weekendingdate"].map(epiweek_label)
    week_df["week_idx"] = (week_df["week"].astype(np.int32) - 1).clip(0, 52)
    return week_df


def build_processed_panel(matrix, observed_mask, fill_sources, week_df, state_codes, state_names):
    records = []
    for state_id, (state_code, state_name) in enumerate(zip(state_codes, state_names)):
        state_df = week_df.copy()
        state_df["state_id"] = state_id
        state_df["state_code"] = state_code
        state_df["state_name"] = state_name
        state_df["weekly_confirmed_influenza_hospital_admissions"] = matrix[:, state_id].astype(np.float32)
        state_df["is_observed"] = observed_mask[:, state_id].astype(np.int8)
        state_df["is_imputed"] = (1 - state_df["is_observed"]).astype(np.int8)
        state_df["imputation_source"] = fill_sources[:, state_id]
        records.append(state_df)
    panel = pd.concat(records, ignore_index=True)
    return panel[
        [
            "week_id",
            "epiweek",
            "week_end_date",
            "week_idx",
            "state_id",
            "state_code",
            "state_name",
            "weekly_confirmed_influenza_hospital_admissions",
            "is_observed",
            "is_imputed",
            "imputation_source",
        ]
    ].copy()


def make_windows(matrix, observed_mask, week_indices, week_df, input_len, output_len):
    features = matrix[:, :, None].astype(np.float32)
    num_weeks = features.shape[0]
    num_samples = num_weeks - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({num_weeks}) for input_len={input_len} and output_len={output_len}"
        )

    xs = []
    ys = []
    x_masks = []
    y_masks = []
    week_idx_xs = []
    week_idx_ys = []
    sample_ranges = []
    epiweeks = week_df["epiweek"].tolist()
    week_end_dates = week_df["week_end_date"].tolist()

    for start_idx in range(num_samples):
        input_end = start_idx + input_len
        target_end = input_end + output_len
        xs.append(features[start_idx:input_end])
        ys.append(features[input_end:target_end])
        x_masks.append(observed_mask[start_idx:input_end, :, None])
        y_masks.append(observed_mask[input_end:target_end, :, None])
        week_idx_xs.append(week_indices[start_idx:input_end])
        week_idx_ys.append(week_indices[input_end:target_end])
        sample_ranges.append(
            {
                "sample_id": start_idx,
                "input_start": epiweeks[start_idx],
                "input_end": epiweeks[input_end - 1],
                "target_start": epiweeks[input_end],
                "target_end": epiweeks[target_end - 1],
                "input_start_date": week_end_dates[start_idx],
                "input_end_date": week_end_dates[input_end - 1],
                "target_start_date": week_end_dates[input_end],
                "target_end_date": week_end_dates[target_end - 1],
            }
        )

    return (
        np.stack(xs),
        np.stack(ys),
        np.stack(x_masks).astype(bool),
        np.stack(y_masks).astype(bool),
        np.stack(week_idx_xs).astype(np.int64),
        np.stack(week_idx_ys).astype(np.int64),
        sample_ranges,
    )


def split_windows(
    xs,
    ys,
    x_masks,
    y_masks,
    week_idx_xs,
    week_idx_ys,
    sample_ranges,
    train_ratio,
    val_ratio,
    output_len,
):
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
            "x_mask": x_masks[start:end],
            "y_mask": y_masks[start:end],
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
            x_mask=payload["x_mask"],
            y_mask=payload["y_mask"],
            week_idx_x=payload["week_idx_x"],
            week_idx_y=payload["week_idx_y"],
        )


def copy_raw_inputs(raw_dir: Path, source_paths):
    for src_path in source_paths:
        dst_path = raw_dir / src_path.name
        if src_path.resolve() == dst_path.resolve():
            continue
        shutil.copy2(src_path, dst_path)


def build_quality_report(matrix, week_df, state_codes, state_names, adj):
    per_state = []
    for state_id, (state_code, state_name) in enumerate(zip(state_codes, state_names)):
        series = matrix[:, state_id]
        per_state.append(
            {
                "state_name": state_name,
                "state_code": state_code,
                "num_weeks": int(len(series)),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "mean_value": float(series.mean()),
            }
        )

    return {
        "num_states": len(state_names),
        "num_weeks": len(week_df),
        "start_epiweek": week_df.iloc[0]["epiweek"],
        "end_epiweek": week_df.iloc[-1]["epiweek"],
        "start_week_end_date": week_df.iloc[0]["week_end_date"],
        "end_week_end_date": week_df.iloc[-1]["week_end_date"],
        "adjacency_shape": list(adj.shape),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_diagonal_all_ones": bool(np.all(np.diag(adj) == 1.0)),
        "per_state": per_state,
    }


def main():
    args = build_parser().parse_args()
    validate_args(args)

    source_csv_path = Path(args.source_csv).resolve()
    if not source_csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv_path}")

    manifest = load_manifest_if_present(source_csv_path)
    grouped_df, state_codes = load_and_clean_source(source_csv_path, args.include_dc)
    weekending_dates, matrix, observed_mask, fill_sources, span_info = build_week_matrix(
        grouped_df,
        state_codes,
        args.start_date,
        args.end_date,
        args.negative_policy,
        args.missing_policy,
    )
    state_names = [US_STATE_NAMES[code] for code in state_codes]
    adj = build_adjacency(state_codes)
    week_df = build_week_index(weekending_dates)
    panel = build_processed_panel(matrix, observed_mask, fill_sources, week_df, state_codes, state_names)
    week_indices = week_df["week_idx"].to_numpy(dtype=np.int32)
    xs, ys, x_masks, y_masks, week_idx_xs, week_idx_ys, sample_ranges = make_windows(
        matrix, observed_mask, week_indices, week_df, args.input_len, args.output_len
    )
    split_data = split_windows(
        xs,
        ys,
        x_masks,
        y_masks,
        week_idx_xs,
        week_idx_ys,
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
    panel.to_csv(processed_dir / "panel.csv", index=False)
    pd.DataFrame(
        {
            "state_id": np.arange(len(state_names), dtype=np.int32),
            "state_name": state_names,
            "state_code": state_codes,
        }
    ).to_csv(processed_dir / "state_index.csv", index=False)
    quality_report = build_quality_report(matrix, week_df, state_codes, state_names, adj)
    with open(processed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    save_npz_splits(package_dir, split_data)
    with open(package_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f)

    train_value_channel = split_data["train"]["x"][..., 0]
    metric_name = "weekly_confirmed_influenza_hospital_admissions"
    meta = {
        "dataset_name": args.dataset_name,
        "source": "CDC NHSN Weekly Hospital Respiratory Data",
        "source_csv": str(source_csv_path),
        "source_manifest": manifest,
        "metric": metric_name,
        "source_column": "totalconfflunewadm",
        "include_dc": args.include_dc,
        "date_start": week_df.iloc[0]["week_end_date"],
        "date_end": week_df.iloc[-1]["week_end_date"],
        "start_epiweek": week_df.iloc[0]["epiweek"],
        "end_epiweek": week_df.iloc[-1]["epiweek"],
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
        "time_index_type": "weekly",
        "week_index_semantics": "iso_week_zero_based_0_52_from_weekendingdate",
        "temporal_feature_names": ["epiweek"],
        "negative_policy": "clip_zero" if args.negative_policy == "clip" else args.negative_policy,
        "missing_policy": args.missing_policy,
        "missing_value_count": int(span_info["missing_value_count"]),
        "adjacency": {
            "type": "state_border",
            "self_loops": True,
            "shape": [int(adj.shape[0]), int(adj.shape[1])],
        },
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
    print(f"Epiweek range: {week_df.iloc[0]['epiweek']} -> {week_df.iloc[-1]['epiweek']}")


if __name__ == "__main__":
    main()
