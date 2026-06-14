import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = [
    "ili_us_states_h4_leakfree",
    "ili_us_states_h8_leakfree",
    "ili_us_states_h12_leakfree",
    "us_states_nhsn_flu_hosp_h4_leakfree",
    "us_states_nhsn_flu_hosp_h8_leakfree",
    "us_states_nhsn_flu_hosp_h12_leakfree",
]


# Approximate state/DC centroids and 2020 Census apportionment populations.
# These are static public geographic covariates, not disease outcomes.
STATE_INFO = {
    "Alabama": (32.806671, -86.791130, 5024279),
    "Alaska": (61.370716, -152.404419, 733391),
    "Arizona": (33.729759, -111.431221, 7151502),
    "Arkansas": (34.969704, -92.373123, 3011524),
    "California": (36.116203, -119.681564, 39538223),
    "Colorado": (39.059811, -105.311104, 5773714),
    "Connecticut": (41.597782, -72.755371, 3605944),
    "Delaware": (39.318523, -75.507141, 989948),
    "District of Columbia": (38.897438, -77.026817, 689545),
    "Florida": (27.766279, -81.686783, 21538187),
    "Georgia": (33.040619, -83.643074, 10711908),
    "Hawaii": (21.094318, -157.498337, 1455271),
    "Idaho": (44.240459, -114.478828, 1839106),
    "Illinois": (40.349457, -88.986137, 12812508),
    "Indiana": (39.849426, -86.258278, 6785528),
    "Iowa": (42.011539, -93.210526, 3190369),
    "Kansas": (38.526600, -96.726486, 2937880),
    "Kentucky": (37.668140, -84.670067, 4505836),
    "Louisiana": (31.169546, -91.867805, 4657757),
    "Maine": (44.693947, -69.381927, 1362359),
    "Maryland": (39.063946, -76.802101, 6177224),
    "Massachusetts": (42.230171, -71.530106, 7029917),
    "Michigan": (43.326618, -84.536095, 10077331),
    "Minnesota": (45.694454, -93.900192, 5706494),
    "Mississippi": (32.741646, -89.678696, 2961279),
    "Missouri": (38.456085, -92.288368, 6154913),
    "Montana": (46.921925, -110.454353, 1084225),
    "Nebraska": (41.125370, -98.268082, 1961504),
    "Nevada": (38.313515, -117.055374, 3104614),
    "New Hampshire": (43.452492, -71.563896, 1377529),
    "New Jersey": (40.298904, -74.521011, 9288994),
    "New Mexico": (34.840515, -106.248482, 2117522),
    "New York": (42.165726, -74.948051, 20201249),
    "North Carolina": (35.630066, -79.806419, 10439388),
    "North Dakota": (47.528912, -99.784012, 779094),
    "Ohio": (40.388783, -82.764915, 11799448),
    "Oklahoma": (35.565342, -96.928917, 3959353),
    "Oregon": (44.572021, -122.070938, 4237256),
    "Pennsylvania": (40.590752, -77.209755, 13002700),
    "Rhode Island": (41.680893, -71.511780, 1097379),
    "South Carolina": (33.856892, -80.945007, 5118425),
    "South Dakota": (44.299782, -99.438828, 886667),
    "Tennessee": (35.747845, -86.692345, 6910840),
    "Texas": (31.054487, -97.563461, 29145505),
    "Utah": (40.150032, -111.862434, 3271616),
    "Vermont": (44.045876, -72.710686, 643077),
    "Virginia": (37.769337, -78.169968, 8631393),
    "Washington": (47.400902, -121.490494, 7705281),
    "West Virginia": (38.491226, -80.954453, 1793716),
    "Wisconsin": (44.268543, -89.616508, 5893718),
    "Wyoming": (42.755966, -107.302490, 576851),
}


def package_dir(dataset_name):
    return Path("dataset") / dataset_name / dataset_name


def load_meta(dataset_name):
    with open(package_dir(dataset_name) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_adj(dataset_name):
    with open(package_dir(dataset_name) / "adj_mx.pkl", "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float32)


def save_adj(path, adj):
    with open(path, "wb") as f:
        pickle.dump(adj.astype(np.float32), f)


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def state_arrays(state_order):
    missing = [state for state in state_order if state not in STATE_INFO]
    if missing:
        raise KeyError(f"Missing state metadata for: {missing}")
    lat = np.array([STATE_INFO[state][0] for state in state_order], dtype=np.float64)
    lon = np.array([STATE_INFO[state][1] for state in state_order], dtype=np.float64)
    pop = np.array([STATE_INFO[state][2] for state in state_order], dtype=np.float64)
    return lat, lon, pop


def distance_matrix(state_order):
    lat, lon, _ = state_arrays(state_order)
    n = len(state_order)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(lat[i], lon[i], lat[j], lon[j])
    return dist


def sym_topk_from_scores(scores, k, larger_is_better=True, weighted=True):
    n = scores.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    k = max(0, min(int(k), n - 1))
    work = scores.copy()
    np.fill_diagonal(work, -np.inf if larger_is_better else np.inf)
    for i in range(n):
        if k == 0:
            continue
        if larger_is_better:
            idx = np.argpartition(-work[i], kth=k - 1)[:k]
        else:
            idx = np.argpartition(work[i], kth=k - 1)[:k]
        if weighted:
            vals = scores[i, idx].astype(np.float32)
            if not larger_is_better:
                vals = 1.0 / np.maximum(vals, 1e-6)
            adj[i, idx] = vals
        else:
            adj[i, idx] = 1.0
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    max_val = adj[~np.eye(n, dtype=bool)].max(initial=0.0)
    if weighted and max_val > 0:
        off_diag = ~np.eye(n, dtype=bool)
        adj[off_diag] = adj[off_diag] / max_val
        np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def build_distance_knn(state_order, k):
    dist = distance_matrix(state_order)
    return sym_topk_from_scores(dist, k=k, larger_is_better=False, weighted=False)


def build_gravity_graph(state_order, k):
    _, _, pop = state_arrays(state_order)
    dist = distance_matrix(state_order)
    scores = (pop[:, None] * pop[None, :]) / np.maximum(dist, 1.0) ** 2
    np.fill_diagonal(scores, 0.0)
    return sym_topk_from_scores(scores, k=k, larger_is_better=True, weighted=True)


def build_correlation_graph(dataset_name, k):
    train_npz = package_dir(dataset_name) / "train.npz"
    x_train = np.load(train_npz)["x"][..., 0]
    num_nodes = x_train.shape[2]
    series_by_node = x_train.transpose(2, 0, 1).reshape(num_nodes, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(series_by_node)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    return sym_topk_from_scores(corr, k=k, larger_is_better=True, weighted=True)


def graph_stats(adj):
    n = adj.shape[0]
    off_diag = adj.copy()
    np.fill_diagonal(off_diag, 0.0)
    undirected_edges = int(np.count_nonzero(np.triu(off_diag > 0, k=1)))
    return {
        "shape": list(adj.shape),
        "symmetric": bool(np.allclose(adj, adj.T)),
        "diag_all_one": bool(np.allclose(np.diag(adj), 1.0)),
        "finite": bool(np.isfinite(adj).all()),
        "undirected_edges": undirected_edges,
        "density_excluding_self": float(np.count_nonzero(off_diag > 0) / (n * (n - 1))),
        "min": float(adj.min()),
        "max": float(adj.max()),
    }


def build_for_dataset(dataset_name, k):
    meta = load_meta(dataset_name)
    state_order = meta["state_order"]
    out_dir = package_dir(dataset_name) / "graph_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "border": load_adj(dataset_name),
        "identity": np.eye(len(state_order), dtype=np.float32),
        f"distance_knn_k{k}": build_distance_knn(state_order, k),
        f"correlation_topk_k{k}": build_correlation_graph(dataset_name, k),
        f"gravity_topk_k{k}": build_gravity_graph(state_order, k),
    }

    report = {}
    for name, adj in variants.items():
        path = out_dir / f"adj_{name}.pkl"
        save_adj(path, adj)
        report[name] = {"path": str(path), **graph_stats(adj)}
    with open(out_dir / "graph_variants_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def write_markdown(all_reports, output_path):
    lines = [
        "# Graph Sensitivity Variant Report",
        "",
        "Variants generated: border, identity, distance-kNN, train-only correlation top-k, and population-distance gravity top-k.",
        "",
        "| Dataset | Variant | Path | Edges | Density | Symmetric | Diag=1 | Finite |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for dataset_name, report in all_reports.items():
        for variant, stats in report.items():
            lines.append(
                f"| {dataset_name} | {variant} | `{stats['path']}` | "
                f"{stats['undirected_edges']} | {stats['density_excluding_self']:.4f} | "
                f"{stats['symmetric']} | {stats['diag_all_one']} | {stats['finite']} |"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build adjacency variants for graph sensitivity experiments.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--k", type=int, default=4, help="top-k neighbors per node before symmetrization")
    parser.add_argument(
        "--report",
        type=str,
        default="review/graph_sensitivity_variant_report.md",
        help="Markdown report path",
    )
    args = parser.parse_args()

    all_reports = {}
    for dataset_name in args.datasets:
        all_reports[dataset_name] = build_for_dataset(dataset_name, args.k)
    write_markdown(all_reports, Path(args.report))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
