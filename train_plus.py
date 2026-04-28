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
    parser.add_argument("--data", type=str, default="bike_drop", help="dataset name")
    parser.add_argument(
        "--model",
        type=str,
        default="st_llm_plus",
        choices=[
            "st_llm_plus",
            "dt_st_llm_plus",
            "epi_st_llm_plus",
            "epi_st_llm_plus_v2b",
            "AR",
            "VAR",
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
        "--stllm_use_llm",
        type=str2bool,
        default=True,
        help="whether ST-LLM+ uses the GPT/PFA branch; set false for the ST-LLM+ w/o LLM ablation",
    )
    parser.add_argument("--n_hidden", type=int, default=64, help="baseline hidden size")
    parser.add_argument("--n_layer", type=int, default=1, help="baseline recurrent layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="baseline dropout")
    parser.add_argument("--rnn_model", type=str, default="GRU", choices=["LSTM", "GRU", "RNN"])
    parser.add_argument("--bi", action="store_true", help="use bidirectional RNN in cola_gnn")
    parser.add_argument("--k", type=int, default=10, help="cola_gnn convolution channels")
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
        choices=["mlp", "cross_attn"],
        help="parameter generator for epi_st_llm_plus beta/gamma heads",
    )
    parser.add_argument(
        "--epi_param_attn_heads",
        type=int,
        default=4,
        help="number of attention heads for cross-attention epi parameter generator",
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


def load_adj_mx(dataset_path):
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
    if args.model in {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus", "epi_st_llm_plus_v2b"}:
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
            )
        return model.to(device)

    from earth_baselines import AR, STGCN, VAR, cola_gnn

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
    llm_family = {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus", "epi_st_llm_plus_v2b"}

    def __init__(self, args, scaler, adj_mx, device, semantic_adj_mx=None):
        self.args = args
        self.scaler = scaler
        self.model = build_model(args, device, adj_mx, semantic_adj_mx)
        self.model.to(device)
        self.device = device
        self.output_is_normalized = args.model not in {"epi_st_llm_plus", "epi_st_llm_plus_v2b"}
        self.is_epi_model = args.model in {"epi_st_llm_plus", "epi_st_llm_plus_v2b"}
        self.stage3_started = False
        self.use_warm_start = args.model == "epi_st_llm_plus" and bool(args.warm_start_ckpt)

        optimizer_name = args.optimizer
        if optimizer_name is None:
            optimizer_name = "ranger" if args.model in {"st_llm_plus", "dt_st_llm_plus", "epi_st_llm_plus"} else "adam"
        if optimizer_name == "ranger":
            self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay
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

    def _compute_pred_loss(self, predict, real):
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

    def _step(self, x, y, temporal_idx_x=None, training=False):
        model_x, model_temporal = self._prepare_input(x, temporal_idx_x)
        if training:
            self.model.train()
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
                if self.args.model in {"st_llm_plus", "dt_st_llm_plus"}
                else self.model(model_x)[0]
            )
            output = self._format_output(output)

        output_for_loss = self._select_prediction_day(output)
        real = self._format_target(y)
        predict = output_for_loss if not self.output_is_normalized else self.scaler.inverse_transform(output_for_loss)

        loss = self._compute_pred_loss(predict, real)
        mass_loss = torch.zeros((), device=predict.device)
        param_loss = torch.zeros((), device=predict.device)
        if self.is_epi_model:
            mass_loss, param_loss = self._compute_epi_regularizers(model_output)
            loss = loss + self.args.lambda_mass * mass_loss + self.args.lambda_param * param_loss

        if training:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

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

    def train(self, x, y, temporal_idx_x=None):
        return self._step(x, y, temporal_idx_x=temporal_idx_x, training=True)

    def eval(self, x, y, temporal_idx_x=None):
        with torch.no_grad():
            return self._step(x, y, temporal_idx_x=temporal_idx_x, training=False)

    def predict(self, x, temporal_idx_x=None):
        self.model.eval()
        with torch.no_grad():
            model_x, model_temporal = self._prepare_input(x, temporal_idx_x)
            if self.is_epi_model:
                output = self.model(model_x, model_temporal)
            else:
                output = (
                    self.model(model_x, model_temporal)
                    if self.args.model in {"st_llm_plus", "dt_st_llm_plus"}
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


def evaluate_testset(engine, dataloader, scaler, device, output_len, target_day=None):
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy[..., 0].transpose(1, 2)

    for _, (x, y, temporal_idx_x, temporal_idx_y) in enumerate(dataloader["test_loader"].get_iterator()):
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
        pred = yhat[:, :, target_idx] if engine.is_epi_model else scaler.inverse_transform(yhat[:, :, 0])
        real = realy[:, :, target_idx]
        return util.metric(pred, real)

    horizon_metrics = []
    for i in range(output_len):
        pred = yhat[:, :, i] if engine.is_epi_model else scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        horizon_metrics.append(util.metric(pred, real))

    return horizon_metrics


def main():
    parser = build_parser()
    args = parser.parse_args()
    seed_it(6666)
    dataset_name = resolve_dataset_config(args)
    resolve_model_config(args)
    adj_mx = load_adj_mx(args.data)
    semantic_adj_mx = None
    if args.model == "dt_st_llm_plus":
        semantic_adj_mx = build_semantic_adj_mx(args.data, args.semantic_graph_top_k)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu to run on CPU.")
    args.cuda = device.type == "cuda"
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
    path = os.path.join(args.save + dataset_name + target_suffix + "_" + args.model + ablation_suffix)

    val_time = []
    train_time = []
    result = []
    test_result = []
    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = Trainer(args, scaler, adj_mx, device, semantic_adj_mx)

    print("start training...", flush=True)
    for i in range(1, args.epochs + 1):
        engine.maybe_enable_joint_tuning(i)

        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        train_mass = []
        train_param = []

        t1 = time.time()
        for _, (x, y, temporal_idx_x, temporal_idx_y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            train_temporal_idx_x = (
                torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
            )
            metrics = engine.train(trainx, trainy, train_temporal_idx_x)
            train_loss.append(metrics["loss"])
            train_mape.append(metrics["mape"])
            train_rmse.append(metrics["rmse"])
            train_wmape.append(metrics["wmape"])
            train_mass.append(metrics["mass_loss"])
            train_param.append(metrics["param_loss"])

        t2 = time.time()
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []
        valid_mass = []
        valid_param = []

        s1 = time.time()
        for _, (x, y, temporal_idx_x, temporal_idx_y) in enumerate(dataloader["val_loader"].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            val_temporal_idx_x = (
                torch.LongTensor(temporal_idx_x).to(device) if temporal_idx_x is not None else None
            )
            metrics = engine.eval(valx, valy, val_temporal_idx_x)
            valid_loss.append(metrics["loss"])
            valid_mape.append(metrics["mape"])
            valid_rmse.append(metrics["rmse"])
            valid_wmape.append(metrics["wmape"])
            valid_mass.append(metrics["mass_loss"])
            valid_param.append(metrics["param_loss"])

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

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")

        if i >= args.min_epochs and epochs_since_best_mae >= args.es_patience:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training ends")
    print("The epoch of the best result", bestid)
    print("The valid loss of the best model", str(round(best_valid_loss, 4)))

    engine.model.load_state_dict(torch.load(os.path.join(path, "best_model.pth"), map_location=device))
    amae = []
    amape = []
    armse = []
    awmape = []

    horizon_metrics = evaluate_testset(
        engine,
        dataloader,
        scaler,
        device,
        args.output_len,
        target_day=args.target_day,
    )

    if args.target_day is not None:
        mae, mape, rmse, wmape = horizon_metrics
        print(
            "Evaluate best model on test data for target day {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                args.target_day, mae, rmse, mape, wmape
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
        return

    for i, metrics in enumerate(horizon_metrics):
        print(
            "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                i + 1, metrics[0], metrics[2], metrics[1], metrics[3]
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


if __name__ == "__main__":
    torch.cuda.empty_cache()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
