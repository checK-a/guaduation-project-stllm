"""
Generate prediction vs ground-truth overview plots for all models on h14 COVID dataset.
Saves one figure per state (or a grid of selected states).
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET = "us_states_covid_jhu_20200301_20230309_ma7_h14"
DATA_DIR = f"dataset/{DATASET}/{DATASET}"
LOGS_DIR = "logs"

MODEL_DIRS = {
    "ST-LLM+": "2026-04-19-23-15-54-us_states_covid_jhu_20200301_20230309_ma7_h14_st_llm_plus",
    "AR":       "2026-04-19-22-46-29-us_states_covid_jhu_20200301_20230309_ma7_h14_AR",
    "VAR":      "2026-04-19-22-46-40-us_states_covid_jhu_20200301_20230309_ma7_h14_VAR",
    "cola_gnn": "2026-04-19-22-46-51-us_states_covid_jhu_20200301_20230309_ma7_h14_cola_gnn",
    "STGCN":    "2026-04-19-23-04-47-us_states_covid_jhu_20200301_20230309_ma7_h14_STGCN",
}

COLORS = {
    "Ground Truth": "#333333",
    "ST-LLM+":      "#e6194b",
    "AR":           "#3cb44b",
    "VAR":          "#4363d8",
    "cola_gnn":     "#f58231",
    "STGCN":        "#911eb4",
}

STATE_NAMES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
]


def load_scaler():
    meta = json.load(open(f"{DATA_DIR}/meta.json"))
    return meta["scaler_mean"], meta["scaler_std"]


def inverse(x, mean, std):
    return x * std + mean


def load_ground_truth(mean, std):
    """Returns (N_samples, 14, 51) in original scale."""
    d = np.load(f"{DATA_DIR}/test.npz")
    y = d["y"][..., 0]          # (N, 14, 51)
    return y                    # already original scale (not normalised in y)


def run_inference(model_name, model_dir, device, mean, std):
    """Load best_model.pth and run inference on test set. Returns (N, 14, 51)."""
    path = os.path.join(LOGS_DIR, model_dir, "best_model.pth")
    d = np.load(f"{DATA_DIR}/test.npz")
    x_raw = d["x"]                          # (N, 24, 51, 1)
    temporal = d.get("temporal_idx_x", None)

    # normalise x[:,0] like training does
    x = x_raw.copy().astype(np.float32)
    x[..., 0] = (x[..., 0] - mean) / std

    import util
    adj_mx = util.load_graph_data(f"dataset/{DATASET}/adj_mx.pkl")

    # build args namespace
    args = SimpleNamespace(
        model=model_name.lower().replace("-", "_").replace("+", "_plus"),
        num_nodes=51, input_len=24, output_len=14, input_dim=1,
        window=24, horizon=14,
        n_hidden=64, n_layer=1, dropout=0.2,
        rnn_model="GRU", bi=False, k=10,
        llm_layer=6, U=1,
        cuda=(device.type == "cuda"),
    )
    # fix model name key
    name_map = {
        "st_llm_plus": "st_llm_plus",
        "ar": "AR", "var": "VAR",
        "cola_gnn": "cola_gnn", "stgcn": "STGCN",
    }
    args.model = {
        "ST-LLM+": "st_llm_plus",
        "AR": "AR", "VAR": "VAR",
        "cola_gnn": "cola_gnn", "STGCN": "STGCN",
    }[model_name]

    from train_plus import build_model
    model = build_model(args, device, adj_mx)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    batch = 64
    N = x.shape[0]
    preds = []
    with torch.no_grad():
        for start in range(0, N, batch):
            xb = torch.tensor(x[start:start+batch]).to(device)
            if args.model == "st_llm_plus":
                xb_in = xb.transpose(1, 3)   # (B, 1, 51, 24)
                tb = None
                if temporal is not None:
                    tb = torch.LongTensor(temporal[start:start+batch]).to(device)
                out = model(xb_in, tb)        # (B, 14, 51, 1)
                out = out.transpose(1, 3)     # (B, 1, 51, 14)  → squeeze below
                out = out.squeeze(1).permute(0, 2, 1)  # (B, 14, 51)
            else:
                xb_in = xb[..., 0]           # (B, 24, 51)
                out, _ = model(xb_in)         # (B, 14, 51)
                out = out.transpose(1, 2)     # (B, 51, 14) → (B, 14, 51)
                out = out.permute(0, 2, 1)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)[:N]   # (N, 14, 51)
    preds = inverse(preds, mean, std)
    return preds


def flatten_time(arr):
    """(N, 14, 51) → (N*14, 51): flatten samples × horizon into a time axis."""
    N, H, S = arr.shape
    return arr.reshape(N * H, S)


def plot_overview(gt_flat, pred_dict, out_path, n_states=12):
    """
    Plot n_states subplots. Each subplot shows the full flattened time series
    for ground truth and all model predictions for one state.
    """
    # pick states spread across the 51
    indices = np.linspace(0, 50, n_states, dtype=int)
    ncols = 3
    nrows = (n_states + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3.2),
                             constrained_layout=True)
    axes = axes.flatten()

    T = gt_flat.shape[0]
    x_axis = np.arange(T)

    for plot_i, state_i in enumerate(indices):
        ax = axes[plot_i]
        ax.plot(x_axis, gt_flat[:, state_i], color=COLORS["Ground Truth"],
                lw=1.5, label="Ground Truth", zorder=5)
        for mname, pred_flat in pred_dict.items():
            ax.plot(x_axis, pred_flat[:, state_i], color=COLORS[mname],
                    lw=1.0, alpha=0.85, label=mname)
        ax.set_title(STATE_NAMES[state_i], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time step (sample × horizon)", fontsize=7)
        ax.set_ylabel("Cases", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("COVID-19 14-day Prediction: All Models vs Ground Truth\n(12 representative US states)",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = load_scaler()
    gt = load_ground_truth(mean, std)   # (N, 14, 51)
    gt_flat = flatten_time(gt)          # (N*14, 51)

    pred_dict = {}
    for mname, mdir in MODEL_DIRS.items():
        print(f"Running inference: {mname} ...")
        try:
            preds = run_inference(mname, mdir, device, mean, std)
            pred_dict[mname] = flatten_time(preds)
            print(f"  {mname} done, shape={preds.shape}")
        except Exception as e:
            print(f"  {mname} FAILED: {e}")

    plot_overview(gt_flat, pred_dict, "prediction_overview.png", n_states=12)


if __name__ == "__main__":
    main()
