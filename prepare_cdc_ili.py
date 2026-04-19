import argparse
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

ABBR_TO_STATE = {abbr: name for name, abbr in STATE_TO_ABBR.items()}
STATE_ALIASES = {
    "WASHINGTON DC": "District of Columbia",
    "WASHINGTON, DC": "District of Columbia",
    "DISTRICT OF COLUMBIA": "District of Columbia",
    "D C": "District of Columbia",
    "DC": "District of Columbia",
    "D.C.": "District of Columbia",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_epiweek(value):
    match = re.fullmatch(r"(\d{4})W(\d{1,2})", value.strip())
    if not match:
        raise ValueError(f"Invalid epiweek format: {value}. Expected YYYYWww.")
    year = int(match.group(1))
    week = int(match.group(2))
    if week < 1 or week > 53:
        raise ValueError(f"Invalid epiweek week number: {value}")
    return year, week


def normalize_header(value):
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_state_name(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    text_upper = normalize_header(text)
    if text_upper in STATE_ALIASES:
        return STATE_ALIASES[text_upper]
    if text_upper in ABBR_TO_STATE:
        return ABBR_TO_STATE[text_upper]

    title = re.sub(r"\s+", " ", text).title()
    if title in STATE_TO_ABBR:
        return title
    return None


def detect_column(df, candidates):
    normalized_to_original = {normalize_header(col): col for col in df.columns}
    for candidate in candidates:
        normalized_candidate = normalize_header(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]
    return None


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_state_order(include_dc):
    states = [state for state in STATE_TO_ABBR if include_dc or state != "District of Columbia"]
    return sorted(states)


def filter_epiweek_range(df, start_epiweek, end_epiweek):
    start_year, start_week = parse_epiweek(start_epiweek)
    end_year, end_week = parse_epiweek(end_epiweek)

    after_start = (df["year"] > start_year) | (
        (df["year"] == start_year) & (df["week"] >= start_week)
    )
    before_end = (df["year"] < end_year) | (
        (df["year"] == end_year) & (df["week"] <= end_week)
    )
    return df.loc[after_start & before_end].copy()


def load_and_clean_ili(ili_csv_path, state_order, start_epiweek, end_epiweek):
    ili_df = pd.read_csv(ili_csv_path)

    region_col = detect_column(ili_df, ["REGION", "REGION NAME", "STATE", "AREA"])
    year_col = detect_column(ili_df, ["YEAR"])
    week_col = detect_column(ili_df, ["WEEK"])
    ili_col = detect_column(
        ili_df,
        [
            "% WEIGHTED ILI",
            "WEIGHTED ILI",
            "PERCENT WEIGHTED ILI",
            "ILI WEIGHTED",
        ],
    )
    region_type_col = detect_column(ili_df, ["REGION TYPE", "REGION_TYPE"])

    missing_columns = [
        name
        for name, col in {
            "REGION": region_col,
            "YEAR": year_col,
            "WEEK": week_col,
            "% WEIGHTED ILI": ili_col,
        }.items()
        if col is None
    ]
    if missing_columns:
        raise ValueError(f"Missing required ILI columns: {missing_columns}")

    if region_type_col is not None:
        region_type = ili_df[region_type_col].astype(str).str.strip().str.lower()
        ili_df = ili_df.loc[region_type.eq("states")].copy()

    ili_df = ili_df.rename(
        columns={
            region_col: "region",
            year_col: "year",
            week_col: "week",
            ili_col: "weighted_ili",
        }
    )
    ili_df["state_name"] = ili_df["region"].map(normalize_state_name)
    ili_df = ili_df.loc[ili_df["state_name"].isin(state_order)].copy()
    if ili_df.empty:
        raise ValueError("No state-level ILI rows remained after filtering.")

    ili_df["year"] = pd.to_numeric(ili_df["year"], errors="coerce")
    ili_df["week"] = pd.to_numeric(ili_df["week"], errors="coerce")
    ili_df["weighted_ili"] = pd.to_numeric(ili_df["weighted_ili"], errors="coerce")
    ili_df = ili_df.dropna(subset=["year", "week", "weighted_ili"])
    ili_df["year"] = ili_df["year"].astype(int)
    ili_df["week"] = ili_df["week"].astype(int)

    ili_df = filter_epiweek_range(ili_df, start_epiweek, end_epiweek)
    if ili_df.empty:
        raise ValueError("No ILI rows remained after epiweek range filtering.")

    ili_df = (
        ili_df.groupby(["state_name", "year", "week"], as_index=False)["weighted_ili"]
        .mean()
        .sort_values(["year", "week", "state_name"])
        .reset_index(drop=True)
    )

    global_weeks = (
        ili_df[["year", "week"]]
        .drop_duplicates()
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )
    global_weeks["epiweek_id"] = np.arange(len(global_weeks), dtype=np.int32)
    global_weeks["epiweek"] = global_weeks.apply(
        lambda row: f"{int(row['year']):04d}W{int(row['week']):02d}", axis=1
    )

    state_index = pd.DataFrame(
        {"state_id": np.arange(len(state_order), dtype=np.int32), "state_name": state_order}
    )
    panel_index = global_weeks.assign(_key=1).merge(state_index.assign(_key=1), on="_key")
    panel_index = panel_index.drop(columns="_key")

    panel = panel_index.merge(ili_df, on=["state_name", "year", "week"], how="left")
    panel["is_imputed"] = panel["weighted_ili"].isna().astype(np.int8)
    panel["weighted_ili"] = (
        panel.groupby("state_name", group_keys=False)["weighted_ili"]
        .apply(lambda series: series.interpolate(method="linear", limit_direction="both"))
        .astype(np.float32)
    )
    if panel["weighted_ili"].isna().any():
        raise ValueError("ILI panel still contains NaNs after interpolation.")

    panel = panel.sort_values(["epiweek_id", "state_id"]).reset_index(drop=True)
    return panel, global_weeks, state_index


def load_and_build_adjacency(adj_csv_path, state_order):
    edge_df = pd.read_csv(adj_csv_path)
    if edge_df.shape[1] < 2:
        raise ValueError("Adjacency CSV must contain at least two columns.")

    state_a_col = detect_column(
        edge_df, ["state", "state1", "source", "from", "region", "state_a"]
    )
    state_b_col = detect_column(
        edge_df, ["neighbor", "state2", "target", "to", "border", "state_b"]
    )
    if state_a_col is None or state_b_col is None:
        state_a_col, state_b_col = edge_df.columns[:2]

    edge_df = edge_df.rename(columns={state_a_col: "state_a", state_b_col: "state_b"})
    edge_df["state_a"] = edge_df["state_a"].map(normalize_state_name)
    edge_df["state_b"] = edge_df["state_b"].map(normalize_state_name)
    edge_df = edge_df.dropna(subset=["state_a", "state_b"])
    edge_df = edge_df.loc[
        edge_df["state_a"].isin(state_order) & edge_df["state_b"].isin(state_order)
    ].copy()
    if edge_df.empty:
        raise ValueError("No usable adjacency edges remained after state normalization.")

    state_to_idx = {state: idx for idx, state in enumerate(state_order)}
    adj = np.zeros((len(state_order), len(state_order)), dtype=np.float32)

    for row in edge_df.itertuples(index=False):
        i = state_to_idx[row.state_a]
        j = state_to_idx[row.state_b]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    np.fill_diagonal(adj, 1.0)
    return adj


def build_feature_tensor(panel, global_weeks, state_order):
    num_weeks = len(global_weeks)
    num_states = len(state_order)

    value_tensor = (
        panel.pivot(index="epiweek_id", columns="state_id", values="weighted_ili")
        .reindex(index=np.arange(num_weeks), columns=np.arange(num_states))
        .to_numpy(dtype=np.float32)
    )

    if np.isnan(value_tensor).any():
        raise ValueError("Value tensor contains NaNs after panel pivot.")

    features = value_tensor[:, :, None].astype(np.float32)
    week_indices = (global_weeks["week"].to_numpy(dtype=np.int32) - 1).clip(0, 52)
    return features, value_tensor, week_indices


def make_windows(features, values, week_indices, global_weeks, input_len, output_len):
    num_weeks = features.shape[0]
    num_samples = num_weeks - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({num_weeks}) for input_len={input_len} and output_len={output_len}"
        )

    xs = []
    ys = []
    week_idx_xs = []
    week_idx_ys = []
    sample_ranges = []
    epiweeks = global_weeks["epiweek"].tolist()

    for start_idx in range(num_samples):
        input_end = start_idx + input_len
        target_end = input_end + output_len
        xs.append(features[start_idx:input_end])
        ys.append(values[input_end:target_end, :, None])
        week_idx_xs.append(week_indices[start_idx:input_end])
        week_idx_ys.append(week_indices[input_end:target_end])
        sample_ranges.append(
            {
                "sample_id": start_idx,
                "input_start": epiweeks[start_idx],
                "input_end": epiweeks[input_end - 1],
                "target_start": epiweeks[input_end],
                "target_end": epiweeks[target_end - 1],
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


def build_quality_report(panel, global_weeks, state_order, adj):
    per_state = []
    for state in state_order:
        state_df = panel.loc[panel["state_name"] == state]
        per_state.append(
            {
                "state_name": state,
                "num_weeks": int(len(state_df)),
                "num_imputed": int(state_df["is_imputed"].sum()),
                "min_ili": float(state_df["weighted_ili"].min()),
                "max_ili": float(state_df["weighted_ili"].max()),
                "mean_ili": float(state_df["weighted_ili"].mean()),
            }
        )

    report = {
        "num_states": len(state_order),
        "num_weeks": len(global_weeks),
        "start_epiweek": global_weeks.iloc[0]["epiweek"],
        "end_epiweek": global_weeks.iloc[-1]["epiweek"],
        "num_imputed_total": int(panel["is_imputed"].sum()),
        "adjacency_shape": list(adj.shape),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_diagonal_all_ones": bool(np.all(np.diag(adj) == 1.0)),
        "per_state": per_state,
    }
    return report


def copy_raw_inputs(raw_dir, ili_csv_path, adj_csv_path):
    for src_path in [Path(ili_csv_path), Path(adj_csv_path)]:
        dst_path = raw_dir / src_path.name
        if src_path.resolve() == dst_path.resolve():
            continue
        shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare CDC ILI dataset for ST-LLM migration.")
    parser.add_argument("--ili_csv", type=str, required=True, help="Path to raw CDC ILINet CSV.")
    parser.add_argument("--adj_csv", type=str, required=True, help="Path to raw adjacency edge CSV.")
    parser.add_argument("--dataset_name", type=str, default="ili_us_states", help="Output dataset name.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset",
        help="Root directory under which the prepared dataset will be written.",
    )
    parser.add_argument("--start_epiweek", type=str, default="2013W40", help="Inclusive start epiweek.")
    parser.add_argument("--end_epiweek", type=str, default="2023W40", help="Inclusive end epiweek.")
    parser.add_argument("--input_len", type=int, default=24, help="Number of input weeks.")
    parser.add_argument("--output_len", type=int, default=4, help="Number of prediction weeks.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--include_dc",
        type=parse_bool,
        default=True,
        help="Whether to include Washington, DC as a node.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input_len and output_len must be positive.")

    ili_csv_path = Path(args.ili_csv).resolve()
    adj_csv_path = Path(args.adj_csv).resolve()
    if not ili_csv_path.exists():
        raise FileNotFoundError(f"ILI CSV not found: {ili_csv_path}")
    if not adj_csv_path.exists():
        raise FileNotFoundError(f"Adjacency CSV not found: {adj_csv_path}")

    dataset_root = Path(args.output_root).resolve() / args.dataset_name
    raw_dir = ensure_dir(dataset_root / "raw")
    processed_dir = ensure_dir(dataset_root / "processed")
    package_dir = ensure_dir(dataset_root / args.dataset_name)

    state_order = build_state_order(args.include_dc)
    panel, global_weeks, state_index = load_and_clean_ili(
        ili_csv_path=ili_csv_path,
        state_order=state_order,
        start_epiweek=args.start_epiweek,
        end_epiweek=args.end_epiweek,
    )
    adj = load_and_build_adjacency(adj_csv_path, state_order)
    features, values, week_indices = build_feature_tensor(panel, global_weeks, state_order)
    xs, ys, week_idx_xs, week_idx_ys, sample_ranges = make_windows(
        features, values, week_indices, global_weeks, args.input_len, args.output_len
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

    copy_raw_inputs(raw_dir, ili_csv_path, adj_csv_path)
    panel.to_csv(processed_dir / "panel.csv", index=False)
    state_index.to_csv(processed_dir / "state_index.csv", index=False)

    quality_report = build_quality_report(panel, global_weeks, state_order, adj)
    with open(processed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    save_npz_splits(package_dir, split_data)
    with open(package_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj, f)

    train_value_channel = split_data["train"]["x"][..., 0]
    meta = {
        "dataset_name": args.dataset_name,
        "start_epiweek": str(global_weeks.iloc[0]["epiweek"]),
        "end_epiweek": str(global_weeks.iloc[-1]["epiweek"]),
        "num_nodes": len(state_order),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "feature_names": ["weighted_ili"],
        "state_order": state_order,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "scaler_mean": float(train_value_channel.mean()),
        "scaler_std": float(train_value_channel.std()),
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
    print(
        f"Epiweek range: {global_weeks.iloc[0]['epiweek']} -> {global_weeks.iloc[-1]['epiweek']}"
    )


if __name__ == "__main__":
    main()
