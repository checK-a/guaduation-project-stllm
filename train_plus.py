import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import util
from ranger21 import Ranger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:180"


def safe_torch_save(state_dict, save_path):
    try:
        torch.save(state_dict, save_path)
    except RuntimeError as exc:
        message = str(exc)
        if "inline_container.cc" not in message and "unexpected pos" not in message:
            raise
        print("torch.save failed with zip serialization; retrying with legacy serialization.", flush=True)
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)


def build_parser():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--seed", type=int, default=6666, help="random seed")
    parser.add_argument("--data", type=str, default="bike_drop", help="dataset name")
    parser.add_argument(
        "--adj_override_path",
        type=str,
        default=None,
        help="optional path to an adjacency matrix pickle; used for graph sensitivity experiments",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="st_llm_plus",
        choices=[
            "st_llm_plus",
            "dt_st_llm_plus",
            "epi_st_llm_plus",
            "epi_st_llm_plus_v2b",
            "GCNGPT",
            "GATGPT",
            "AR",
            "VAR",
            "Persistence",
            "GRU",
            "LSTM",
            "CausalGNN",
            "DCRNN",
            "EpiGNN",
            "EpiGNNLite",
            "SIR",
            "SEIR",
            "PatchTST",
            "cola_gnn",
            "STGCN",
        ],
        help="model name",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lrate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="max training epochs")
    parser.add_argument("--input_dim", type=int, default=3, help="input dimension")
    parser.add_argument("--num_nodes", type=int, default=250, help="number of nodes")
    parser.add_argument("--input_len", type=int, default=12, help="history length")
    parser.add_argument("--output_len", type=int, default=12, help="prediction length")
    parser.add_argument(
        "--target_day",
        type=int,
        default=None,
        help="1-based target day for direct single-day prediction; e.g. 14 means predict only day 14",
    )
    parser.add_argument("--llm_layer", type=int, default=6, help="llm layer")
    parser.add_argument("--U", type=int, default=1, help="unfrozen layer")
    parser.add_argument(
        "--llm_graph_injection_layers",
        type=int,
        default=None,
        help=(
            "number of final GPT/PFA layers that receive graph attention bias; "
            "defaults to U when unset"
        ),
    )
    parser.add_argument(
        "--stllm_use_llm",
        type=str2bool,
        default=True,
        help="whether ST-LLM+ uses the GPT/PFA branch; set false for the ST-LLM+ w/o LLM ablation",
    )
    parser.add_argument("--n_hidden", type=int, default=64, help="baseline hidden size")
    parser.add_argument("--n_layer", type=int, default=1, help="baseline recurrent layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="baseline dropout")
    parser.add_argument("--rnn_model", type=str, default="GRU", choices=["LSTM", "GRU", "RNN"])
    parser.add_argument(
        "--dcrnn_filter_type",
        type=str,
        default="laplacian",
        choices=["laplacian", "random_walk", "dual_random_walk"],
        help="diffusion filter type for the EARTH PyTorch DCRNN baseline",
    )
    parser.add_argument("--bi", action="store_true", help="use bidirectional RNN in cola_gnn")
    parser.add_argument("--k", type=int, default=10, help="cola_gnn convolution channels")
    parser.add_argument("--causal_top_k", type=int, default=8, help="top-k directed parents for CausalGNN")
    parser.add_argument("--causal_gnn_layers", type=int, default=2, help="CausalGNN message-passing layers")
    parser.add_argument(
        "--causal_graph_alpha_init",
        type=float,
        default=0.0,
        help="initial logit for static-vs-causal graph fusion in CausalGNN",
    )
    parser.add_argument("--epignn_k", type=int, default=8, help="EpiGNN multi-scale convolution kernels")
    parser.add_argument("--epignn_hidA", type=int, default=64, help="EpiGNN global transmission attention hidden size")
    parser.add_argument("--epignn_hidP", type=int, default=1, help="EpiGNN adaptive pooling height")
    parser.add_argument("--epignn_gcn_layers", type=int, default=2, help="EpiGNN GCN layer count")
    parser.add_argument("--epignn_highway_window", type=int, default=0, help="EpiGNN autoregressive highway window")
    parser.add_argument(
        "--epignn_residual_concat",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether EpiGNN concatenates all GCN layer states like the optional residual branch",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["ranger", "adam", None],
        help="optimizer; defaults to ranger for ST-LLM families and adam for baselines",
    )
    parser.add_argument("--print_every", type=int, default=50, help="")
    parser.add_argument("--wdecay", type=float, default=0.0001, help="weight decay rate")
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "-",
        help="save path",
    )
    parser.add_argument(
        "--save_epoch_checkpoints",
        type=str,
        default="",
        help="comma-separated 1-based epochs to save as epoch_XXX_model.pth",
    )
    parser.add_argument(
        "--eval_checkpoint_epoch",
        type=int,
        default=None,
        help="load epoch_XXX_model.pth for final test evaluation instead of best_model.pth",
    )
    parser.add_argument(
        "--eval_conformal_intervals",
        type=str2bool,
        default=False,
        help="after loading the evaluation checkpoint, calibrate split-conformal intervals on val residuals and evaluate on test",
    )
    parser.add_argument(
        "--conformal_coverages",
        type=str,
        default="0.9,0.95",
        help="comma-separated target coverages for conformal intervals, e.g. 0.9,0.95",
    )
    parser.add_argument(
        "--profile_resources",
        type=str2bool,
        default=False,
        help="write structured runtime, throughput, parameter-count, and GPU-memory statistics",
    )
    parser.add_argument(
        "--resource_report_name",
        type=str,
        default="resource_report.csv",
        help="filename for the per-run resource profile written inside the run log directory",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=100,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="minimum number of epochs to train before early stopping can trigger",
    )
    parser.add_argument(
        "--warm_start_ckpt",
        type=str,
        default=None,
        help="checkpoint path for loading a trained st_llm_plus encoder into epi_st_llm_plus",
    )
    parser.add_argument("--compartment_dim", type=int, default=16, help="latent compartment size")
    parser.add_argument("--lambda_wmape", type=float, default=0.1, help="weight for WMAPE term")
    parser.add_argument("--lambda_mass", type=float, default=0.01, help="weight for mass regularizer")
    parser.add_argument("--lambda_param", type=float, default=0.01, help="weight for parameter smoothness")
    parser.add_argument(
        "--eval_sir_diagnostics",
        type=str2bool,
        default=False,
        help="write latent SIR mass-conservation diagnostics for epi models after loading the evaluation checkpoint",
    )
    parser.add_argument(
        "--sir_diagnostic_splits",
        type=str,
        default="train,val,test",
        help="comma-separated splits for --eval_sir_diagnostics",
    )
    parser.add_argument(
        "--test_y_mask_policy",
        type=str,
        default="point",
        choices=["point", "drop_sample", "drop_node", "drop_sample_or_node"],
        help=(
            "how to apply y_mask during final test metrics: point masks only missing labels, "
            "drop_sample removes any test sample with a missing target, drop_node removes nodes "
            "with any missing test target, and drop_sample_or_node applies both"
        ),
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="full",
        choices=["full", "no_mech", "mech_only", "no_llm", "fixed_params"],
        help="Epi-ST-LLM+ ablation mode; only used for epi models",
    )
    parser.add_argument(
        "--llm_fusion_mode",
        type=str,
        default=None,
        choices=["direct", "none", "residual_gate"],
        help="LLM fusion mode for epi models; defaults to direct for epi_st_llm_plus and residual_gate for epi_st_llm_plus_v2b",
    )
    parser.add_argument(
        "--epi_param_generator",
        type=str,
        default="mlp",
        choices=["mlp", "cross_attn", "temporal_cross_attn"],
        help="parameter generator for epi_st_llm_plus beta/gamma heads",
    )
    parser.add_argument(
        "--epi_param_attn_heads",
        type=int,
        default=4,
        help="number of attention heads for cross-attention epi parameter generator",
    )
    parser.add_argument(
        "--epi_encoder_type",
        type=str,
        default="llm",
        choices=["llm", "transformer"],
        help="encoder used inside epi_st_llm_plus; transformer replaces the GPT/PFA branch for LLM ablation",
    )
    parser.add_argument(
        "--epi_llm_init",
        type=str,
        default="pretrained",
        choices=["pretrained", "random"],
        help="GPT-2 initialization for epi_st_llm_plus; random keeps the architecture but removes language pretraining",
    )
    parser.add_argument(
        "--epi_lora_mode",
        type=str,
        default="lora",
        choices=["lora", "none"],
        help="whether to attach LoRA adapters to the GPT-2 branch in epi_st_llm_plus",
    )
    parser.add_argument(
        "--epi_freeze_gpt",
        type=str2bool,
        default=False,
        help="freeze all GPT-2 branch parameters; useful with --epi_lora_mode none for frozen-backbone ablations",
    )
    parser.add_argument(
        "--epi_graph_mode",
        type=str,
        default="adjacency",
        choices=["adjacency", "identity"],
        help=(
            "legacy graph mode for epi_st_llm_plus; used for both LLM graph bias and "
            "mechanism propagation unless the more specific graph-mode args are set"
        ),
    )
    parser.add_argument(
        "--epi_llm_graph_mode",
        type=str,
        default=None,
        choices=["adjacency", "identity"],
        help="graph mode for the Epi-ST-LLM+ LLM/PFA encoder only",
    )
    parser.add_argument(
        "--epi_mech_graph_mode",
        type=str,
        default=None,
        choices=["adjacency", "identity"],
        help="graph mode for the Epi-ST-LLM+ mechanism rollout only",
    )
    parser.add_argument(
        "--epi_use_temporal_gate",
        type=str2bool,
        default=True,
        help="whether temporal_cross_attn uses learnable temporal residual gates; false fixes both temporal gates to zero",
    )
    parser.add_argument(
        "--epi_temporal_gate_mode",
        type=str,
        default=None,
        choices=["learnable", "zero", "one"],
        help=(
            "temporal_cross_attn residual gate mode: learnable uses sigmoid gate init -1.0, "
            "zero removes temporal mixing, and one directly adds the temporal branch"
        ),
    )
    parser.add_argument(
        "--epi_temporal_gate_init",
        type=float,
        default=-1.0,
        help="initial logit for learnable temporal residual gates in temporal_cross_attn",
    )
    parser.add_argument(
        "--temporal_patch_len",
        type=int,
        default=4,
        help="temporal patch length for epi_st_llm_plus_v2b",
    )
    parser.add_argument(
        "--temporal_patch_stride",
        type=int,
        default=4,
        help="temporal patch stride for epi_st_llm_plus_v2b; must equal temporal_patch_len in V2b",
    )
    parser.add_argument(
        "--graph_bias_mode",
        type=str,
        default=None,
        choices=["patch_graph_bias", "none"],
        help="graph attention bias mode for epi_st_llm_plus_v2b",
    )
    parser.add_argument(
        "--graph_bias_scale_init",
        type=float,
        default=1.0,
        help="initial scale for graph attention bias in epi_st_llm_plus_v2b",
    )
    parser.add_argument(
        "--dt_graph_mode",
        type=str,
        default="static_dynamic",
        choices=["static", "dynamic", "static_dynamic", "static_semantic_dynamic"],
        help="graph mode for dt_st_llm_plus",
    )
    parser.add_argument("--dynamic_graph_top_k", type=int, default=5, help="top-k outgoing dynamic graph edges")
    parser.add_argument("--semantic_graph_top_k", type=int, default=8, help="top-k semantic graph edges")
    parser.add_argument(
        "--dynamic_graph_alpha_init",
        type=float,
        default=1.0,
        help="initial logit for static-vs-dynamic fusion in dt_st_llm_plus",
    )
    return parser


def _load_meta_dataset_config(dataset_name):
    meta_path = Path("dataset") / dataset_name / dataset_name / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_config(args):
    dataset_name = args.data
    args.data = f"dataset//{dataset_name}//{dataset_name}"

    meta = _load_meta_dataset_config(dataset_name)
    if meta is not None:
        args.num_nodes = int(meta.get("num_nodes", args.num_nodes))
        args.input_len = int(meta.get("input_len", args.input_len))
        args.input_dim = len(meta.get("feature_names", [])) or args.input_dim
        full_output_len = int(meta.get("output_len", args.output_len))
    elif dataset_name in {"bike_drop", "bike_pick"}:
        args.num_nodes = 250
        full_output_len = args.output_len
    elif dataset_name in {"taxi_drop", "taxi_pick"}:
        args.num_nodes = 266
        full_output_len = args.output_len
    else:
        full_output_len = args.output_len

    args.full_output_len = full_output_len
    if args.target_day is not None:
        if not (1 <= args.target_day <= args.full_output_len):
            raise ValueError(
                f"--target_day must be in [1, {args.full_output_len}] for dataset {dataset_name}"
            )
        args.output_len = (
            args.full_output_len if args.model in {"epi_st_llm_plus", "epi_st_llm_plus_v2b"} else 1
        )
    else:
        args.output_len = args.full_output_len

    args.window = args.input_len
    args.horizon = args.output_len
    return dataset_name


def resolve_model_config(args):
    if args.model == "epi_st_llm_plus_v2b":
        if args.llm_fusion_mode is None:
            args.llm_fusion_mode = "residual_gate"
        if args.graph_bias_mode is None:
            args.graph_bias_mode = "patch_graph_bias"
    elif args.model == "epi_st_llm_plus":
        if args.llm_fusion_mode is None:
            args.llm_fusion_mode = "direct"
        args.graph_bias_mode = None
    else:
        if args.llm_fusion_mode is None:
            args.llm_fusion_mode = "direct"
        args.graph_bias_mode = None


def load_adj_mx(dataset_path, adj_override_path=None):
    if adj_override_path:
        return util.load_graph_data(adj_override_path)
    return util.load_graph_data(f"{dataset_path}/adj_mx.pkl")


def build_semantic_adj_mx(dataset_path, top_k):
    if top_k <= 0:
        return None

    train_npz = Path(dataset_path) / "train.npz"
    if not train_npz.exists():
        return None

    x_train = np.load(train_npz)["x"][..., 0]
    num_nodes = x_train.shape[2]
    series_by_node = x_train.transpose(2, 0, 1).reshape(num_nodes, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(series_by_node)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0).astype(np.float32)
    np.fill_diagonal(corr, 0.0)

    semantic_adj = np.zeros_like(corr, dtype=np.float32)
    k = max(0, min(int(top_k), num_nodes - 1))
    if k > 0:
        top_indices = np.argpartition(-corr, kth=k - 1, axis=1)[:, :k]
        row_indices = np.arange(num_nodes)[:, None]
        semantic_adj[row_indices, top_indices] = corr[row_indices, top_indices]
    np.fill_diagonal(semantic_adj, 1.0)
    return semantic_adj


def build_model(args, device, adj_mx, semantic_adj_mx=None):
    if args.model in {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus", "epi_st_llm_plus_v2b", "GCNGPT", "GATGPT"}:
        if args.model == "GCNGPT":
            from model_GCNGPT import GCNGPT

            model = GCNGPT(
                device=device,
                adj_mx=adj_mx,
                input_dim=args.input_dim,
                channels=args.n_hidden,
                num_nodes=args.num_nodes,
                input_len=args.input_len,
                output_len=args.output_len,
                dropout=args.dropout,
            )
            return model.to(device)
        if args.model == "GATGPT":
            from model_GATGPT import GATGPT

            model = GATGPT(
                device=device,
                adj_mx=adj_mx,
                input_dim=args.input_dim,
                channels=args.n_hidden,
                num_nodes=args.num_nodes,
                input_len=args.input_len,
                output_len=args.output_len,
                dropout=args.dropout,
            )
            return model.to(device)

        from model_ST_LLM_plus import DynamicTransmissionSTLLM, EpiSTLLMPlus, EpiSTLLMPlusV2b, ST_LLM

        if args.model == "st_llm_plus":
            model = ST_LLM(
                device,
                adj_mx,
                args.input_dim,
                args.num_nodes,
                args.input_len,
                args.output_len,
                args.llm_layer,
                args.U,
                args.stllm_use_llm,
            )
        elif args.model == "dt_st_llm_plus":
            model = DynamicTransmissionSTLLM(
                device,
                adj_mx,
                semantic_adj_mx,
                args.input_dim,
                args.num_nodes,
                args.input_len,
                args.output_len,
                args.llm_layer,
                args.U,
                args.stllm_use_llm,
                args.dt_graph_mode,
                args.dynamic_graph_top_k,
                args.semantic_graph_top_k,
                args.dynamic_graph_alpha_init,
            )
        elif args.model == "epi_st_llm_plus_v2b":
            model = EpiSTLLMPlusV2b(
                device,
                adj_mx,
                args.input_dim,
                args.num_nodes,
                args.input_len,
                args.output_len,
                args.llm_layer,
                args.U,
                args.compartment_dim,
                args.ablation_mode,
                args.llm_fusion_mode,
                args.epi_param_generator,
                args.epi_param_attn_heads,
                args.temporal_patch_len,
                args.temporal_patch_stride,
                args.graph_bias_mode,
                args.graph_bias_scale_init,
            )
        else:
            model = EpiSTLLMPlus(
                device,
                adj_mx,
                args.input_dim,
                args.num_nodes,
                args.input_len,
                args.output_len,
                args.llm_layer,
                args.U,
                args.compartment_dim,
                args.ablation_mode,
                args.llm_fusion_mode,
                args.epi_param_generator,
                args.epi_param_attn_heads,
                getattr(args, "epi_encoder_type", "llm"),
                getattr(args, "epi_graph_mode", "adjacency"),
                getattr(args, "epi_llm_graph_mode", None),
                getattr(args, "epi_mech_graph_mode", None),
                getattr(args, "llm_graph_injection_layers", None),
                getattr(args, "epi_use_temporal_gate", True),
                getattr(args, "epi_temporal_gate_mode", None),
                getattr(args, "epi_temporal_gate_init", -1.0),
                getattr(args, "epi_llm_init", "pretrained"),
                getattr(args, "epi_lora_mode", "lora"),
                getattr(args, "epi_freeze_gpt", False),
            )
        return model.to(device)

    from earth_baselines import (
        AR,
        CausalGNN,
        DCRNNModel,
        EpiGNNLite,
        EpiGNNModel,
        GRUBaseline,
        LSTMBaseline,
        PatchTST,
        Persistence,
        SEIRBaseline,
        SIRBaseline,
        STGCN,
        VAR,
        cola_gnn,
    )

    baseline_data = SimpleNamespace(
        m=args.num_nodes,
        d=args.input_dim,
        adj=torch.tensor(adj_mx, dtype=torch.float32),
        orig_adj=torch.tensor(adj_mx, dtype=torch.float32),
    )

    if args.model == "AR":
        model = AR(args, baseline_data)
    elif args.model == "VAR":
        model = VAR(args, baseline_data)
    elif args.model == "Persistence":
        model = Persistence(args, baseline_data)
    elif args.model == "GRU":
        model = GRUBaseline(args, baseline_data)
    elif args.model == "LSTM":
        model = LSTMBaseline(args, baseline_data)
    elif args.model == "CausalGNN":
        model = CausalGNN(args, baseline_data)
    elif args.model == "DCRNN":
        model = DCRNNModel(args, baseline_data)
    elif args.model == "EpiGNN":
        model = EpiGNNModel(args, baseline_data)
    elif args.model == "EpiGNNLite":
        model = EpiGNNLite(args, baseline_data)
    elif args.model == "SIR":
        model = SIRBaseline(args, baseline_data)
    elif args.model == "SEIR":
        model = SEIRBaseline(args, baseline_data)
    elif args.model == "PatchTST":
        model = PatchTST(args, baseline_data)
    elif args.model == "cola_gnn":
        model = cola_gnn(args, baseline_data)
    elif args.model == "STGCN":
        model = STGCN(
            args,
            baseline_data,
            num_nodes=args.num_nodes,
            num_features=args.input_dim,
            num_timesteps_input=args.input_len,
            num_timesteps_output=args.output_len,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model.to(device)


class Trainer:
    llm_family = {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus", "epi_st_llm_plus_v2b", "GCNGPT", "GATGPT"}
    raw_output_models = {"SIR", "SEIR"}
    direct_llm_call_models = {"st_llm_plus", "dt_st_llm_plus", "GCNGPT", "GATGPT"}

    def __init__(self, args, scaler, adj_mx, device, semantic_adj_mx=None):
        self.args = args
        self.scaler = scaler
        self.args.scaler_mean = scaler.mean
        self.args.scaler_std = scaler.std
        self.model = build_model(args, device, adj_mx, semantic_adj_mx)
        self.model.to(device)
        self.device = device
        self.output_is_normalized = args.model not in {
            "epi_st_llm_plus",
            "epi_st_llm_plus_v2b",
            *self.raw_output_models,
        }
        self.is_epi_model = args.model in {"epi_st_llm_plus", "epi_st_llm_plus_v2b"}
        self.stage3_started = False
        self.use_warm_start = args.model == "epi_st_llm_plus" and bool(args.warm_start_ckpt)

        optimizer_name = args.optimizer
        if optimizer_name is None:
            optimizer_name = "ranger" if args.model in {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus", "GCNGPT", "GATGPT"} else "adam"
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            self.optimizer = None
        elif optimizer_name == "ranger":
            self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        else:
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=args.lrate, weight_decay=args.wdecay
            )

        self.clip = 5
        if self.is_epi_model:
            if self.use_warm_start:
                missing, unexpected = self.model.load_encoder_state(args.warm_start_ckpt)
                self.model.freeze_encoder_for_stage2()
                print(
                    "Loaded warm-start encoder weights. "
                    f"Missing keys after partial load: {len(missing)}, unexpected keys ignored: {len(unexpected)}"
                )
            else:
                self.stage3_started = True
                print("No warm-start checkpoint provided. Training epi_st_llm_plus from cold start.")

        print("The number of parameters: {}".format(self.param_num()))
        print("The number of trainable parameters: {}".format(self.count_trainable_params()))
        print(self.model)

    def maybe_enable_joint_tuning(self, epoch_index):
        if not self.is_epi_model or not self.use_warm_start or self.stage3_started:
            return
        joint_tune_epoch = max(2, self.args.epochs // 2 + 1)
        if epoch_index >= joint_tune_epoch:
            self.model.enable_joint_tuning_stage3()
            self.stage3_started = True
            print(
                "Switching epi_st_llm_plus to stage 3 joint tuning at epoch {}. "
                "LoRA and the last PFGA layer are now trainable.".format(epoch_index)
            )

    def param_num(self):
        return sum(param.nelement() for param in self.model.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _prepare_input(self, x, temporal_idx_x=None):
        if self.args.model in self.llm_family:
            model_x = x.transpose(1, 3)
            model_temporal = temporal_idx_x
        else:
            model_x = x[..., 0]
            model_temporal = None
        return model_x, model_temporal

    def _format_output(self, output):
        if self.args.model in self.llm_family:
            return output.transpose(1, 3)
        return output.transpose(1, 2).unsqueeze(1)

    def _select_target_day(self, y):
        if self.args.target_day is None:
            return y
        target_idx = self.args.target_day - 1
        return y[:, target_idx : target_idx + 1, :, :]

    def _select_prediction_day(self, prediction):
        if self.args.target_day is None or prediction.size(-1) == 1:
            return prediction
        target_idx = self.args.target_day - 1
        return prediction[..., target_idx : target_idx + 1]

    def _format_target(self, y):
        y = self._select_target_day(y)
        return y[..., 0].transpose(1, 2).unsqueeze(1)

    def _format_target_mask(self, y_mask):
        if y_mask is None:
            return None
        y_mask = self._select_target_day(y_mask)
        return y_mask[..., 0].transpose(1, 2).unsqueeze(1).bool()

    def _compute_pred_loss(self, predict, real, target_mask=None):
        if target_mask is not None:
            mask = target_mask.to(device=predict.device, dtype=predict.dtype)
            if self.is_epi_model and self.args.target_day is None and predict.size(-1) > 1:
                horizon_weights = torch.linspace(
                    2.0,
                    1.0,
                    steps=predict.size(-1),
                    device=predict.device,
                    dtype=predict.dtype,
                ).view(1, 1, 1, -1)
                horizon_weights = horizon_weights / horizon_weights.mean()
                weights = horizon_weights * mask
            else:
                weights = mask
            denom = weights.sum().clamp_min(1e-6)
            abs_error = torch.abs(real - predict)
            mae = (abs_error * weights).sum() / denom
            wmape = (abs_error * weights).sum() / torch.sum(torch.abs(real) * weights).clamp_min(1e-6)
            return mae + self.args.lambda_wmape * wmape

        if self.is_epi_model and self.args.target_day is None and predict.size(-1) > 1:
            horizon_weights = torch.linspace(
                2.0,
                1.0,
                steps=predict.size(-1),
                device=predict.device,
                dtype=predict.dtype,
            ).view(1, 1, 1, -1)
            horizon_weights = horizon_weights / horizon_weights.mean()
            abs_error = torch.abs(real - predict)
            mae = (abs_error * horizon_weights).mean()
            wmape = torch.sum(abs_error * horizon_weights) / torch.sum(
                torch.abs(real) * horizon_weights
            ).clamp_min(1e-6)
            return mae + self.args.lambda_wmape * wmape

        mae = util.MAE_torch(predict, real, 0.0)
        wmape = util.WMAPE_torch(predict, real, 0.0)
        return mae + self.args.lambda_wmape * wmape

    def _compute_epi_regularizers(self, model_output):
        if model_output.get("skip_mech_regularizers", False):
            zero = torch.zeros((), device=self.device)
            return zero, zero

        beta = model_output["beta"]
        gamma = model_output["gamma"]
        s0 = model_output["s0"]
        i0 = model_output["i0"]
        r0 = model_output["r0"]
        s_states = model_output["S"]
        i_states = model_output["I"]
        r_states = model_output["R"]

        initial_mass = (s0 + i0 + r0).mean(dim=-1)
        rollout_mass = (s_states + i_states + r_states).mean(dim=-1)
        mass_loss = torch.abs(rollout_mass - initial_mass.unsqueeze(1)).mean()

        if beta.size(1) > 1:
            beta_smooth = torch.abs(beta[:, 1:] - beta[:, :-1]).mean()
            gamma_smooth = torch.abs(gamma[:, 1:] - gamma[:, :-1]).mean()
            param_loss = beta_smooth + gamma_smooth
        else:
            param_loss = torch.zeros((), device=beta.device)

        return mass_loss, param_loss

    def _step(self, x, y, temporal_idx_x=None, y_mask=None, training=False):
        model_x, model_temporal = self._prepare_input(x, temporal_idx_x)
        if training:
            self.model.train()
            if self.optimizer is not None:
                self.optimizer.zero_grad()
        else:
            self.model.eval()

        if self.is_epi_model:
            model_output = self.model(model_x, model_temporal, return_aux=True)
            output = self._format_output(model_output["prediction"])
        else:
            model_output = None
            output = (
                self.model(model_x, model_temporal)
                if self.args.model in self.direct_llm_call_models
                else self.model(model_x)[0]
            )
            output = self._format_output(output)

        output_for_loss = self._select_prediction_day(output)
        real = self._format_target(y)
        target_mask = self._format_target_mask(y_mask)
        predict = output_for_loss if not self.output_is_normalized else self.scaler.inverse_transform(output_for_loss)

        loss = self._compute_pred_loss(predict, real, target_mask)
        mass_loss = torch.zeros((), device=predict.device)
        param_loss = torch.zeros((), device=predict.device)
        if self.is_epi_model:
            mass_loss, param_loss = self._compute_epi_regularizers(model_output)
            loss = loss + self.args.lambda_mass * mass_loss + self.args.lambda_param * param_loss

        if training and self.optimizer is not None:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        if target_mask is not None:
            mape = util.MAPE_masked_torch(predict, real, target_mask).item()
            rmse = util.RMSE_masked_torch(predict, real, target_mask).item()
            wmape = util.WMAPE_masked_torch(predict, real, target_mask).item()
        else:
            mape = util.MAPE_torch(predict, real, 0.0).item()
            rmse = util.RMSE_torch(predict, real, 0.0).item()
            wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return {
            "loss": loss.item(),
            "mape": mape,
            "rmse": rmse,
            "wmape": wmape,
            "mass_loss": mass_loss.item(),
            "param_loss": param_loss.item(),
        }

    def train(self, x, y, temporal_idx_x=None, y_mask=None):
        return self._step(x, y, temporal_idx_x=temporal_idx_x, y_mask=y_mask, training=True)

    def eval(self, x, y, temporal_idx_x=None, y_mask=None):
        with torch.no_grad():
            return self._step(x, y, temporal_idx_x=temporal_idx_x, y_mask=y_mask, training=False)

    def predict(self, x, temporal_idx_x=None):
        self.model.eval()
        with torch.no_grad():
            model_x, model_temporal = self._prepare_input(x, temporal_idx_x)
            if self.is_epi_model:
                output = self.model(model_x, model_temporal)
            else:
                output = (
                    self.model(model_x, model_temporal)
                    if self.args.model in self.direct_llm_call_models
                    else self.model(model_x)[0]
                )
            return self._format_output(output)


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def apply_y_mask_policy(mask, policy):
    if mask is None or policy == "point":
        return mask
    if mask.dim() not in {2, 3}:
        raise ValueError(f"Expected a 2D or 3D y mask, got shape {tuple(mask.shape)}")

    adjusted = mask.bool()
    if policy in {"drop_sample", "drop_sample_or_node"}:
        reduce_dims = tuple(range(1, adjusted.dim()))
        sample_keep = adjusted.all(dim=reduce_dims)
        view_shape = [adjusted.size(0)] + [1] * (adjusted.dim() - 1)
        adjusted = adjusted & sample_keep.view(*view_shape)

    if policy in {"drop_node", "drop_sample_or_node"}:
        if adjusted.dim() == 2:
            node_keep = adjusted.all(dim=0)
            adjusted = adjusted & node_keep.view(1, -1)
        else:
            node_keep = adjusted.all(dim=(0, 2))
            adjusted = adjusted & node_keep.view(1, -1, 1)

    return adjusted


def evaluate_testset(engine, dataloader, scaler, device, output_len, target_day=None, y_mask_policy="point"):
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy[..., 0].transpose(1, 2)
    realy_mask = None
    if "y_mask_test" in dataloader:
        realy_mask = torch.BoolTensor(dataloader["y_mask_test"]).to(device)
        realy_mask = realy_mask[..., 0].transpose(1, 2)
        realy_mask = apply_y_mask_policy(realy_mask, y_mask_policy)

    for _, (x, y, temporal_idx_x, temporal_idx_y, x_mask, y_mask) in enumerate(
        dataloader["test_loader"].get_iterator_with_masks()
    ):
        testx = torch.Tensor(x).to(device)
        test_temporal_idx_x = (
            torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
        )
        preds = engine.predict(testx, test_temporal_idx_x)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    if target_day is not None:
        target_idx = target_day - 1
        if engine.is_epi_model:
            pred = yhat[:, :, target_idx]
        else:
            pred = yhat[:, :, 0]
            if engine.output_is_normalized:
                pred = scaler.inverse_transform(pred)
        real = realy[:, :, target_idx]
        if realy_mask is not None:
            mask = realy_mask[:, :, target_idx]
            return util.metric_masked(pred, real, mask)
        return util.metric(pred, real)

    horizon_metrics = []
    for i in range(output_len):
        pred = yhat[:, :, i] if not engine.output_is_normalized else scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        if realy_mask is not None:
            horizon_metrics.append(util.metric_masked(pred, real, realy_mask[:, :, i]))
        else:
            horizon_metrics.append(util.metric(pred, real))

    return horizon_metrics


def collect_predictions_for_split(engine, dataloader, scaler, device, split_name, target_day=None):
    loader = dataloader[f"{split_name}_loader"]
    y_array = torch.Tensor(dataloader[f"y_{split_name}"]).to(device)
    real = y_array[..., 0].transpose(1, 2)
    real_mask = None
    if f"y_mask_{split_name}" in dataloader:
        real_mask = torch.BoolTensor(dataloader[f"y_mask_{split_name}"]).to(device)
        real_mask = real_mask[..., 0].transpose(1, 2)
    else:
        real_mask = torch.ones_like(real, dtype=torch.bool)

    outputs = []
    for _, (x, y, temporal_idx_x, temporal_idx_y, x_mask, y_mask) in enumerate(
        loader.get_iterator_with_masks()
    ):
        split_x = torch.Tensor(x).to(device)
        split_temporal_idx_x = (
            torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
        )
        preds = engine.predict(split_x, split_temporal_idx_x)
        outputs.append(preds.squeeze(1))

    pred = torch.cat(outputs, dim=0)
    pred = pred[: real.size(0), ...]

    if target_day is not None:
        target_idx = target_day - 1
        if engine.is_epi_model:
            pred = pred[:, :, target_idx : target_idx + 1]
        else:
            pred = pred[:, :, :1]
            if engine.output_is_normalized:
                pred = scaler.inverse_transform(pred)
        real = real[:, :, target_idx : target_idx + 1]
        real_mask = real_mask[:, :, target_idx : target_idx + 1]
    elif engine.output_is_normalized:
        pred = scaler.inverse_transform(pred)

    return pred.detach().cpu().numpy(), real.detach().cpu().numpy(), real_mask.detach().cpu().numpy().astype(bool)


def conformal_quantile(residuals, alpha):
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return np.nan
    q_rank = int(np.ceil((residuals.size + 1) * (1.0 - alpha)))
    q_rank = min(max(q_rank, 1), residuals.size)
    return float(np.partition(residuals, q_rank - 1)[q_rank - 1])


def compute_conformal_intervals(val_pred, val_real, val_mask, test_pred, test_real, test_mask, coverages):
    rows = []
    horizon_count = test_pred.shape[-1]
    for coverage in coverages:
        alpha = 1.0 - coverage
        for horizon_idx in range(horizon_count):
            val_valid = val_mask[:, :, horizon_idx]
            test_valid = test_mask[:, :, horizon_idx]

            residuals = np.abs(
                val_real[:, :, horizon_idx][val_valid]
                - val_pred[:, :, horizon_idx][val_valid]
            )
            qhat = conformal_quantile(residuals, alpha)
            if not np.isfinite(qhat):
                rows.append(
                    {
                        "coverage_target": coverage,
                        "horizon": horizon_idx + 1,
                        "qhat": np.nan,
                        "empirical_coverage": np.nan,
                        "mpiw": np.nan,
                        "winkler": np.nan,
                        "n_calibration": int(residuals.size),
                        "n_test": int(test_valid.sum()),
                    }
                )
                continue

            pred_h = test_pred[:, :, horizon_idx]
            real_h = test_real[:, :, horizon_idx]
            lower = np.maximum(pred_h - qhat, 0.0)
            upper = pred_h + qhat

            valid_lower = lower[test_valid]
            valid_upper = upper[test_valid]
            valid_real = real_h[test_valid]
            inside = (valid_real >= valid_lower) & (valid_real <= valid_upper)
            width = valid_upper - valid_lower
            below = valid_real < valid_lower
            above = valid_real > valid_upper
            winkler = width.copy()
            winkler[below] += (2.0 / alpha) * (valid_lower[below] - valid_real[below])
            winkler[above] += (2.0 / alpha) * (valid_real[above] - valid_upper[above])

            rows.append(
                {
                    "coverage_target": coverage,
                    "horizon": horizon_idx + 1,
                    "qhat": qhat,
                    "empirical_coverage": float(np.mean(inside)) if inside.size else np.nan,
                    "mpiw": float(np.mean(width)) if width.size else np.nan,
                    "winkler": float(np.mean(winkler)) if winkler.size else np.nan,
                    "n_calibration": int(residuals.size),
                    "n_test": int(test_valid.sum()),
                }
            )

        horizon_rows = [row for row in rows if row["coverage_target"] == coverage and isinstance(row["horizon"], int)]
        valid_rows = [row for row in horizon_rows if np.isfinite(row["empirical_coverage"])]
        rows.append(
            {
                "coverage_target": coverage,
                "horizon": "avg",
                "qhat": float(np.nanmean([row["qhat"] for row in valid_rows])) if valid_rows else np.nan,
                "empirical_coverage": float(np.nanmean([row["empirical_coverage"] for row in valid_rows])) if valid_rows else np.nan,
                "mpiw": float(np.nanmean([row["mpiw"] for row in valid_rows])) if valid_rows else np.nan,
                "winkler": float(np.nanmean([row["winkler"] for row in valid_rows])) if valid_rows else np.nan,
                "n_calibration": int(np.sum([row["n_calibration"] for row in valid_rows])),
                "n_test": int(np.sum([row["n_test"] for row in valid_rows])),
            }
        )
    return rows


def parse_coverages(value):
    coverages = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        coverage = float(item)
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"Conformal coverage must be between 0 and 1, got {coverage}")
        coverages.append(coverage)
    if not coverages:
        raise ValueError("At least one conformal coverage is required.")
    return coverages


def evaluate_conformal_intervals(engine, dataloader, scaler, device, coverages, output_path, target_day=None):
    val_pred, val_real, val_mask = collect_predictions_for_split(
        engine, dataloader, scaler, device, "val", target_day=target_day
    )
    test_pred, test_real, test_mask = collect_predictions_for_split(
        engine, dataloader, scaler, device, "test", target_day=target_day
    )
    rows = compute_conformal_intervals(
        val_pred,
        val_real,
        val_mask,
        test_pred,
        test_real,
        test_mask,
        coverages,
    )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Conformal interval results written to {output_path}")
    for row in rows:
        print(
            "Conformal coverage {:.2f}, horizon {}, qhat: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}, Winkler: {:.4f}, n_cal/n_test: {}/{}".format(
                row["coverage_target"],
                row["horizon"],
                row["qhat"],
                row["empirical_coverage"],
                row["mpiw"],
                row["winkler"],
                row["n_calibration"],
                row["n_test"],
            )
        )
    return df


def parse_split_list(value):
    splits = [item.strip() for item in str(value).split(",") if item.strip()]
    valid = {"train", "val", "test"}
    unknown = [split for split in splits if split not in valid]
    if unknown:
        raise ValueError(f"Unknown diagnostic split(s): {unknown}. Valid choices are {sorted(valid)}")
    if not splits:
        raise ValueError("At least one diagnostic split is required.")
    return splits


def _summarize_tensor(tensor, prefix):
    flat = tensor.detach().float().reshape(-1)
    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_std": flat.std(unbiased=False).item(),
        f"{prefix}_min": flat.min().item(),
        f"{prefix}_max": flat.max().item(),
    }


def _sir_batch_diagnostics(model_output, eps=1e-6):
    if model_output.get("skip_mech_regularizers", False):
        return None

    s0 = model_output["s0"].detach()
    i0 = model_output["i0"].detach()
    r0 = model_output["r0"].detach()
    s_states = model_output["S"].detach()
    i_states = model_output["I"].detach()
    r_states = model_output["R"].detach()
    delta_inf = model_output["delta_inf"].detach()
    delta_rec = model_output["delta_rec"].detach()

    initial_mass = (s0 + i0 + r0).mean(dim=-1)
    rollout_mass = (s_states + i_states + r_states).mean(dim=-1)
    mass_drift = rollout_mass - initial_mass.unsqueeze(1)
    abs_mass_drift = torch.abs(mass_drift)
    rel_mass_drift = abs_mass_drift / initial_mass.unsqueeze(1).abs().clamp_min(eps)

    s_prev = torch.cat([s0.unsqueeze(1), s_states[:, :-1]], dim=1)
    i_prev = torch.cat([i0.unsqueeze(1), i_states[:, :-1]], dim=1)
    inf_clip = delta_inf >= (s_prev - eps)
    rec_clip = delta_rec >= (i_prev + delta_inf - eps)

    compartment_min = torch.minimum(torch.minimum(s_states.min(), i_states.min()), r_states.min())
    init_mass_flat = initial_mass.reshape(-1)
    rollout_mass_flat = rollout_mass.reshape(-1)
    return {
        "n_windows": int(s_states.size(0)),
        "mass_abs_drift_mean": abs_mass_drift.mean().item(),
        "mass_abs_drift_max": abs_mass_drift.max().item(),
        "mass_rel_drift_mean": rel_mass_drift.mean().item(),
        "mass_rel_drift_max": rel_mass_drift.max().item(),
        "mass_signed_drift_mean": mass_drift.mean().item(),
        "initial_mass_mean": init_mass_flat.mean().item(),
        "initial_mass_std": init_mass_flat.std(unbiased=False).item(),
        "rollout_mass_mean": rollout_mass_flat.mean().item(),
        "rollout_mass_std": rollout_mass_flat.std(unbiased=False).item(),
        "compartment_min": compartment_min.item(),
        "negative_compartment_count": int(
            (s_states < -eps).sum().item()
            + (i_states < -eps).sum().item()
            + (r_states < -eps).sum().item()
        ),
        "infection_clip_ratio": inf_clip.float().mean().item(),
        "recovery_clip_ratio": rec_clip.float().mean().item(),
        "delta_inf_mean": delta_inf.float().mean().item(),
        "delta_rec_mean": delta_rec.float().mean().item(),
        "delta_inf_max": delta_inf.float().max().item(),
        "delta_rec_max": delta_rec.float().max().item(),
        **_summarize_tensor(model_output["beta"].detach(), "beta"),
        **_summarize_tensor(model_output["gamma"].detach(), "gamma"),
        **_summarize_tensor(model_output["y_mech"].detach(), "y_mech"),
        **_summarize_tensor(model_output["y_res"].detach(), "y_res"),
        **_summarize_tensor(model_output["prediction"].detach(), "prediction"),
    }


def evaluate_sir_diagnostics(engine, dataloader, device, splits, output_path):
    if not engine.is_epi_model:
        print("Skipping SIR diagnostics because the selected model is not an epi model.")
        return None

    rows = []
    engine.model.eval()
    with torch.no_grad():
        for split in splits:
            loader = dataloader[f"{split}_loader"]
            batch_rows = []
            split_sample_count = int(dataloader[f"x_{split}"].shape[0])
            seen_samples = 0
            for _, (x, y, temporal_idx_x, temporal_idx_y, x_mask, y_mask) in enumerate(
                loader.get_iterator_with_masks()
            ):
                valid_count = min(x.shape[0], split_sample_count - seen_samples)
                if valid_count <= 0:
                    break
                x = x[:valid_count]
                if temporal_idx_x is not None:
                    temporal_idx_x = temporal_idx_x[:valid_count]
                seen_samples += valid_count

                split_x = torch.Tensor(x).to(device)
                split_temporal_idx_x = (
                    torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
                )
                model_x, model_temporal = engine._prepare_input(split_x, split_temporal_idx_x)
                model_output = engine.model(model_x, model_temporal, return_aux=True)
                row = _sir_batch_diagnostics(model_output)
                if row is not None:
                    batch_rows.append(row)

            if not batch_rows:
                rows.append({"split": split, "n_batches": 0, "n_windows": 0})
                continue

            total_windows = sum(row["n_windows"] for row in batch_rows)
            summary = {"split": split, "n_batches": len(batch_rows), "n_windows": total_windows}
            keys = [key for key in batch_rows[0] if key != "n_windows"]
            for key in keys:
                values = np.array([row[key] for row in batch_rows], dtype=np.float64)
                weights = np.array([row["n_windows"] for row in batch_rows], dtype=np.float64)
                if key == "negative_compartment_count":
                    summary[key] = int(np.sum(values))
                elif key.endswith("_max"):
                    summary[key] = float(np.max(values))
                elif key.endswith("_min") or key == "compartment_min":
                    summary[key] = float(np.min(values))
                else:
                    summary[key] = float(np.average(values, weights=weights))
            rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Latent SIR diagnostics written to {output_path}")
    for row in rows:
        print(
            "SIR diagnostics split={}, mean |mass drift|={:.6e}, max |mass drift|={:.6e}, rel drift={:.6e}, inf_clip={:.4f}, rec_clip={:.4f}".format(
                row["split"],
                row.get("mass_abs_drift_mean", np.nan),
                row.get("mass_abs_drift_max", np.nan),
                row.get("mass_rel_drift_mean", np.nan),
                row.get("infection_clip_ratio", np.nan),
                row.get("recovery_clip_ratio", np.nan),
            )
        )
    return df


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_memory_stats_mb(device):
    if device.type != "cuda":
        return {
            "peak_gpu_allocated_mb": np.nan,
            "peak_gpu_reserved_mb": np.nan,
            "final_gpu_allocated_mb": np.nan,
            "final_gpu_reserved_mb": np.nan,
        }
    return {
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device.index) / (1024**2),
        "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device.index) / (1024**2),
        "final_gpu_allocated_mb": torch.cuda.memory_allocated(device.index) / (1024**2),
        "final_gpu_reserved_mb": torch.cuda.memory_reserved(device.index) / (1024**2),
    }


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_resource_report(report, output_path):
    output_path = Path(output_path)
    row = {key: json_ready(value) for key, value in report.items()}
    pd.DataFrame([row]).to_csv(output_path, index=False)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    print(f"Resource profile written to {output_path}")


def build_resource_report(
    args,
    dataset_name,
    run_dir,
    dataloader,
    engine,
    device,
    train_time,
    val_time,
    best_epoch,
    best_valid_loss,
    run_start_time,
    test_eval_time,
    conformal_eval_time,
):
    train_samples = int(dataloader["x_train"].shape[0])
    val_samples = int(dataloader["x_val"].shape[0])
    test_samples = int(dataloader["x_test"].shape[0])
    avg_train_time = float(np.mean(train_time)) if train_time else np.nan
    avg_val_time = float(np.mean(val_time)) if val_time else np.nan
    params = int(engine.param_num())
    trainable_params = int(engine.count_trainable_params())

    report = {
        "dataset": dataset_name,
        "model": args.model,
        "run_dir": run_dir,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "adj_override_path": args.adj_override_path or "",
        "target_day": args.target_day if args.target_day is not None else "",
        "num_nodes": args.num_nodes,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "input_dim": args.input_dim,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "min_epochs": args.min_epochs,
        "es_patience": args.es_patience,
        "epochs_completed": len(train_time),
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_valid_loss": best_valid_loss,
        "total_params": params,
        "trainable_params": trainable_params,
        "trainable_param_ratio": trainable_params / params if params else np.nan,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_batches": dataloader["train_loader"].num_batch,
        "val_batches": dataloader["val_loader"].num_batch,
        "test_batches": dataloader["test_loader"].num_batch,
        "avg_train_sec_per_epoch": avg_train_time,
        "std_train_sec_per_epoch": float(np.std(train_time)) if train_time else np.nan,
        "avg_val_sec_per_epoch": avg_val_time,
        "std_val_sec_per_epoch": float(np.std(val_time)) if val_time else np.nan,
        "total_train_loop_sec": float(np.sum(train_time)) if train_time else np.nan,
        "total_val_loop_sec": float(np.sum(val_time)) if val_time else np.nan,
        "test_eval_sec": test_eval_time,
        "conformal_eval_sec": conformal_eval_time,
        "total_wall_clock_sec": time.time() - run_start_time,
        "train_samples_per_sec": train_samples / avg_train_time if avg_train_time and np.isfinite(avg_train_time) else np.nan,
        "val_samples_per_sec": val_samples / avg_val_time if avg_val_time and np.isfinite(avg_val_time) else np.nan,
        "test_samples_per_sec": test_samples / test_eval_time if test_eval_time and np.isfinite(test_eval_time) else np.nan,
        "profile_resources": args.profile_resources,
        "eval_conformal_intervals": args.eval_conformal_intervals,
    }
    report.update(gpu_memory_stats_mb(device))
    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_start_time = time.time()
    seed_it(args.seed)
    dataset_name = resolve_dataset_config(args)
    resolve_model_config(args)
    if args.epi_temporal_gate_mode is None:
        args.epi_temporal_gate_mode = "learnable" if args.epi_use_temporal_gate else "zero"
    if args.epi_llm_graph_mode is None:
        args.epi_llm_graph_mode = args.epi_graph_mode
    if args.epi_mech_graph_mode is None:
        args.epi_mech_graph_mode = args.epi_graph_mode
    adj_mx = load_adj_mx(args.data, args.adj_override_path)
    semantic_adj_mx = None
    if args.model == "dt_st_llm_plus":
        semantic_adj_mx = build_semantic_adj_mx(args.data, args.semantic_graph_top_k)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu to run on CPU.")
    args.cuda = device.type == "cuda"
    if args.profile_resources and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]

    best_valid_loss = float("inf")
    bestid = None
    epochs_since_best_mae = 0
    target_suffix = f"_d{args.target_day}" if args.target_day is not None else ""
    ablation_suffix = ""
    if args.model == "st_llm_plus":
        ablation_suffix = "_full" if args.stllm_use_llm else "_no_llm"
    elif args.model == "dt_st_llm_plus":
        ablation_suffix = "_" + args.dt_graph_mode
        if not args.stllm_use_llm:
            ablation_suffix += "_no_llm"
    elif args.model == "epi_st_llm_plus":
        ablation_suffix = (
            "_"
            + args.ablation_mode
            + "_"
            + args.llm_fusion_mode
            + "_param_"
            + args.epi_param_generator
        )
        if args.epi_encoder_type != "llm":
            ablation_suffix += "_encoder_" + args.epi_encoder_type
        else:
            if args.epi_llm_init != "pretrained":
                ablation_suffix += "_llm_init_" + args.epi_llm_init
            if args.epi_lora_mode != "lora":
                ablation_suffix += "_lora_" + args.epi_lora_mode
            if args.epi_freeze_gpt:
                ablation_suffix += "_freeze_gpt"
        if args.epi_llm_graph_mode == args.epi_mech_graph_mode and args.epi_llm_graph_mode != "adjacency":
            ablation_suffix += "_graph_" + args.epi_llm_graph_mode
        else:
            if args.epi_llm_graph_mode != "adjacency":
                ablation_suffix += "_llm_graph_" + args.epi_llm_graph_mode
            if args.epi_mech_graph_mode != "adjacency":
                ablation_suffix += "_mech_graph_" + args.epi_mech_graph_mode
        if args.epi_temporal_gate_mode == "zero":
            ablation_suffix += "_no_temporal_gate"
        elif args.epi_temporal_gate_mode == "one":
            ablation_suffix += "_temporal_gate_one"
        elif args.epi_temporal_gate_mode == "learnable" and args.epi_param_generator == "temporal_cross_attn":
            if abs(args.epi_temporal_gate_init - (-1.0)) > 1e-12:
                gate_init = str(args.epi_temporal_gate_init).replace("-", "m").replace(".", "p")
                ablation_suffix += "_temporal_gate_init_" + gate_init
        if args.adj_override_path:
            graph_name = Path(args.adj_override_path).stem.replace("adj_", "")
            ablation_suffix += "_adj_" + graph_name
    elif args.model == "epi_st_llm_plus_v2b":
        ablation_suffix = (
            "_"
            + args.ablation_mode
            + f"_p{args.temporal_patch_len}"
            + "_"
            + args.llm_fusion_mode
            + "_"
            + args.graph_bias_mode
            + "_param_"
            + args.epi_param_generator
        )
    if args.test_y_mask_policy != "point":
        ablation_suffix += "_testmask_" + args.test_y_mask_policy
    path = os.path.join(args.save + dataset_name + target_suffix + "_" + args.model + ablation_suffix)

    val_time = []
    train_time = []
    result = []
    test_result = []
    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    save_epoch_checkpoints = set()
    if args.save_epoch_checkpoints:
        save_epoch_checkpoints = {
            int(epoch.strip())
            for epoch in args.save_epoch_checkpoints.split(",")
            if epoch.strip()
        }
    if args.eval_checkpoint_epoch is not None:
        save_epoch_checkpoints.add(args.eval_checkpoint_epoch)

    engine = Trainer(args, scaler, adj_mx, device, semantic_adj_mx)
    test_eval_time = np.nan
    conformal_eval_time = np.nan

    print("start training...", flush=True)
    for i in range(1, args.epochs + 1):
        engine.maybe_enable_joint_tuning(i)

        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        train_mass = []
        train_param = []

        sync_if_cuda(device)
        t1 = time.time()
        for _, (x, y, temporal_idx_x, temporal_idx_y, x_mask, y_mask) in enumerate(
            dataloader["train_loader"].get_iterator_with_masks()
        ):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            train_temporal_idx_x = (
                torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
            )
            train_y_mask = torch.BoolTensor(y_mask).to(device) if y_mask is not None else None
            metrics = engine.train(trainx, trainy, train_temporal_idx_x, train_y_mask)
            train_loss.append(metrics["loss"])
            train_mape.append(metrics["mape"])
            train_rmse.append(metrics["rmse"])
            train_wmape.append(metrics["wmape"])
            train_mass.append(metrics["mass_loss"])
            train_param.append(metrics["param_loss"])

        sync_if_cuda(device)
        t2 = time.time()
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []
        valid_mass = []
        valid_param = []

        sync_if_cuda(device)
        s1 = time.time()
        for _, (x, y, temporal_idx_x, temporal_idx_y, x_mask, y_mask) in enumerate(
            dataloader["val_loader"].get_iterator_with_masks()
        ):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            val_temporal_idx_x = (
                torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
            )
            val_y_mask = torch.BoolTensor(y_mask).to(device) if y_mask is not None else None
            metrics = engine.eval(valx, valy, val_temporal_idx_x, val_y_mask)
            valid_loss.append(metrics["loss"])
            valid_mape.append(metrics["mape"])
            valid_rmse.append(metrics["rmse"])
            valid_wmape.append(metrics["wmape"])
            valid_mass.append(metrics["mass_loss"])
            valid_param.append(metrics["param_loss"])

        sync_if_cuda(device)
        s2 = time.time()
        print("Epoch: {:03d}, Inference Time: {:.4f} secs".format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mass = np.mean(train_mass)
        mtrain_param = np.mean(train_param)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mass = np.mean(valid_mass)
        mvalid_param = np.mean(valid_param)

        print("-----------------------")
        train_m = pd.Series(
            dict(
                train_loss=mtrain_loss,
                train_rmse=mtrain_rmse,
                train_mape=mtrain_mape,
                train_wmape=mtrain_wmape,
                train_mass=mtrain_mass,
                train_param=mtrain_param,
                valid_loss=mvalid_loss,
                valid_rmse=mvalid_rmse,
                valid_mape=mvalid_mape,
                valid_wmape=mvalid_wmape,
                valid_mass=mvalid_mass,
                valid_param=mvalid_param,
            )
        )
        result.append(train_m)

        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, Train Mass: {:.4f}, Train Param: {:.4f}".format(
                i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape, mtrain_mass, mtrain_param
            ),
            flush=True,
        )
        print(
            "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}, Valid Mass: {:.4f}, Valid Param: {:.4f}".format(
                i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape, mvalid_mass, mvalid_param
            ),
            flush=True,
        )

        if mvalid_loss < best_valid_loss:
            print("###Update tasks appear###")
            best_valid_loss = mvalid_loss
            safe_torch_save(engine.model.state_dict(), os.path.join(path, "best_model.pth"))
            bestid = i
            epochs_since_best_mae = 0
            print("Updating! Valid Loss:{:.4f}, epoch: {}".format(mvalid_loss, i))
        else:
            epochs_since_best_mae += 1
            print("No update")

        if i in save_epoch_checkpoints:
            epoch_ckpt_path = os.path.join(path, f"epoch_{i:03d}_model.pth")
            safe_torch_save(engine.model.state_dict(), epoch_ckpt_path)
            print(f"Saved epoch checkpoint: {epoch_ckpt_path}", flush=True)

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")

        if i >= args.min_epochs and epochs_since_best_mae >= args.es_patience:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training ends")
    print("The epoch of the best result", bestid)
    print("The valid loss of the best model", str(round(best_valid_loss, 4)))

    eval_checkpoint_label = "best model"
    eval_checkpoint_path = os.path.join(path, "best_model.pth")
    if args.eval_checkpoint_epoch is not None:
        eval_checkpoint_label = f"epoch {args.eval_checkpoint_epoch}"
        eval_checkpoint_path = os.path.join(path, f"epoch_{args.eval_checkpoint_epoch:03d}_model.pth")
    print(f"Loading {eval_checkpoint_label} checkpoint from {eval_checkpoint_path}")
    engine.model.load_state_dict(torch.load(eval_checkpoint_path, map_location=device))
    amae = []
    amape = []
    armse = []
    awmape = []

    sync_if_cuda(device)
    test_eval_start = time.time()
    horizon_metrics = evaluate_testset(
        engine,
        dataloader,
        scaler,
        device,
        args.output_len,
        target_day=args.target_day,
        y_mask_policy=args.test_y_mask_policy,
    )
    sync_if_cuda(device)
    test_eval_time = time.time() - test_eval_start

    if args.target_day is not None:
        mae, mape, rmse, wmape = horizon_metrics
        print(
            "Evaluate {} on test data for target day {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                eval_checkpoint_label, args.target_day, mae, rmse, mape, wmape
            )
        )

        test_result.append(
            pd.Series(
                dict(
                    test_loss=mae,
                    test_rmse=rmse,
                    test_mape=mape,
                    test_wmape=wmape,
                )
            )
        )

        test_csv = pd.DataFrame(test_result)
        test_csv.round(8).to_csv(f"{path}/test.csv")
        if args.eval_conformal_intervals:
            sync_if_cuda(device)
            conformal_start = time.time()
            evaluate_conformal_intervals(
                engine,
                dataloader,
                scaler,
                device,
                parse_coverages(args.conformal_coverages),
                f"{path}/conformal_intervals.csv",
                target_day=args.target_day,
            )
            sync_if_cuda(device)
            conformal_eval_time = time.time() - conformal_start
        if args.eval_sir_diagnostics:
            evaluate_sir_diagnostics(
                engine,
                dataloader,
                device,
                parse_split_list(args.sir_diagnostic_splits),
                f"{path}/sir_diagnostics.csv",
            )
        if args.profile_resources:
            report = build_resource_report(
                args,
                dataset_name,
                path,
                dataloader,
                engine,
                device,
                train_time,
                val_time,
                bestid,
                best_valid_loss,
                run_start_time,
                test_eval_time,
                conformal_eval_time,
            )
            write_resource_report(report, os.path.join(path, args.resource_report_name))
        return

    for i, metrics in enumerate(horizon_metrics):
        print(
            "Evaluate {} on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                eval_checkpoint_label, i + 1, metrics[0], metrics[2], metrics[1], metrics[3]
            )
        )

        test_m = pd.Series(
            dict(
                test_loss=np.mean(metrics[0]),
                test_rmse=np.mean(metrics[2]),
                test_mape=np.mean(metrics[1]),
                test_wmape=np.mean(metrics[3]),
            )
        )
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
            args.output_len, np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
        )
    )

    test_result.append(
        pd.Series(
            dict(
                test_loss=np.mean(amae),
                test_rmse=np.mean(armse),
                test_mape=np.mean(amape),
                test_wmape=np.mean(awmape),
            )
        )
    )

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")
    if args.eval_conformal_intervals:
        sync_if_cuda(device)
        conformal_start = time.time()
        evaluate_conformal_intervals(
            engine,
            dataloader,
            scaler,
            device,
            parse_coverages(args.conformal_coverages),
            f"{path}/conformal_intervals.csv",
            target_day=args.target_day,
        )
        sync_if_cuda(device)
        conformal_eval_time = time.time() - conformal_start
    if args.eval_sir_diagnostics:
        evaluate_sir_diagnostics(
            engine,
            dataloader,
            device,
            parse_split_list(args.sir_diagnostic_splits),
            f"{path}/sir_diagnostics.csv",
        )
    if args.profile_resources:
        report = build_resource_report(
            args,
            dataset_name,
            path,
            dataloader,
            engine,
            device,
            train_time,
            val_time,
            bestid,
            best_valid_loss,
            run_start_time,
            test_eval_time,
            conformal_eval_time,
        )
        write_resource_report(report, os.path.join(path, args.resource_report_name))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
