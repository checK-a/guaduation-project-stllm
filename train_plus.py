import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import util
from ranger21 import Ranger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:180"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--data", type=str, default="bike_drop", help="dataset name")
    parser.add_argument(
        "--model",
        type=str,
        default="st_llm_plus",
        choices=["st_llm_plus", "AR", "VAR", "cola_gnn", "STGCN"],
        help="model name",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lrate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="max training epochs")
    parser.add_argument("--input_dim", type=int, default=3, help="input dimension")
    parser.add_argument("--num_nodes", type=int, default=250, help="number of nodes")
    parser.add_argument("--input_len", type=int, default=12, help="history length")
    parser.add_argument("--output_len", type=int, default=12, help="prediction length")
    parser.add_argument("--llm_layer", type=int, default=6, help="llm layer")
    parser.add_argument("--U", type=int, default=1, help="unfrozen layer")
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
        help="optimizer; defaults to ranger for st_llm_plus and adam for baselines",
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
    return parser


def resolve_dataset_config(args):
    dataset_name = args.data
    args.data = f"dataset//{dataset_name}//{dataset_name}"

    if dataset_name in {"bike_drop", "bike_pick"}:
        args.num_nodes = 250
    elif dataset_name in {"taxi_drop", "taxi_pick"}:
        args.num_nodes = 266
    elif dataset_name == "ili_us_states":
        args.num_nodes = 51
        args.input_len = 24
        args.output_len = 4
        args.input_dim = 1

    args.window = args.input_len
    args.horizon = args.output_len
    return dataset_name


def load_adj_mx(dataset_path):
    return util.load_graph_data(f"{dataset_path}/adj_mx.pkl")


def build_model(args, device, adj_mx):
    if args.model == "st_llm_plus":
        from model_ST_LLM_plus import ST_LLM

        model = ST_LLM(
            device,
            adj_mx,
            args.input_dim,
            args.num_nodes,
            args.input_len,
            args.output_len,
            args.llm_layer,
            args.U,
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
    def __init__(self, args, scaler, adj_mx, device):
        self.args = args
        self.model = build_model(args, device, adj_mx)
        self.model.to(device)
        optimizer_name = args.optimizer
        if optimizer_name is None:
            optimizer_name = "ranger" if args.model == "st_llm_plus" else "adam"
        if optimizer_name == "ranger":
            self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay
            )
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.param_num()))
        print("The number of trainable parameters: {}".format(self.count_trainable_params()))
        print(self.model)

    def param_num(self):
        return sum(param.nelement() for param in self.model.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _prepare_input(self, x, week_idx_x=None):
        if self.args.model == "st_llm_plus":
            model_x = x.transpose(1, 3)
            model_week = week_idx_x
        else:
            model_x = x[..., 0]
            model_week = None
        return model_x, model_week

    def _format_output(self, output):
        if self.args.model == "st_llm_plus":
            return output.transpose(1, 3)
        return output.transpose(1, 2).unsqueeze(1)

    def _format_target(self, y):
        return y[..., 0].transpose(1, 2).unsqueeze(1)

    def _step(self, x, y, week_idx_x=None, training=False):
        model_x, model_week = self._prepare_input(x, week_idx_x)
        if training:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        output = self.model(model_x, model_week) if self.args.model == "st_llm_plus" else self.model(model_x)[0]
        if self.args.model == "st_llm_plus":
            output = self._format_output(output)
        else:
            output = self._format_output(output)

        real = self._format_target(y)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)

        if training:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def train(self, x, y, week_idx_x=None):
        return self._step(x, y, week_idx_x=week_idx_x, training=True)

    def eval(self, x, y, week_idx_x=None):
        with torch.no_grad():
            return self._step(x, y, week_idx_x=week_idx_x, training=False)

    def predict(self, x, week_idx_x=None):
        self.model.eval()
        with torch.no_grad():
            model_x, model_week = self._prepare_input(x, week_idx_x)
            output = self.model(model_x, model_week) if self.args.model == "st_llm_plus" else self.model(model_x)[0]
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


def evaluate_testset(engine, dataloader, scaler, device, output_len):
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy[..., 0].transpose(1, 2)

    for _, (x, y, week_idx_x, week_idx_y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        test_week_idx_x = (
            torch.LongTensor(week_idx_x).to(device) if week_idx_x is not None else None
        )
        preds = engine.predict(testx, test_week_idx_x)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    horizon_metrics = []
    for i in range(output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        horizon_metrics.append(util.metric(pred, real))

    return horizon_metrics


def main():
    parser = build_parser()
    args = parser.parse_args()
    seed_it(6666)
    dataset_name = resolve_dataset_config(args)
    adj_mx = load_adj_mx(args.data)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu to run on CPU.")
    args.cuda = device.type == "cuda"
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]

    best_valid_loss = float("inf")
    bestid = None
    epochs_since_best_mae = 0
    path = os.path.join(args.save + dataset_name + "_" + args.model)

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []
    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = Trainer(args, scaler, adj_mx, device)

    print("start training...", flush=True)
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for _, (x, y, week_idx_x, week_idx_y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            train_week_idx_x = (
                torch.LongTensor(week_idx_x).to(device) if week_idx_x is not None else None
            )
            metrics = engine.train(trainx, trainy, train_week_idx_x)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

        t2 = time.time()
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for _, (x, y, week_idx_x, week_idx_y) in enumerate(dataloader["val_loader"].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            val_week_idx_x = (
                torch.LongTensor(week_idx_x).to(device) if week_idx_x is not None else None
            )
            metrics = engine.eval(valx, valy, val_week_idx_x)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()
        print("Epoch: {:03d}, Inference Time: {:.4f} secs".format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        train_m = pd.Series(
            dict(
                train_loss=mtrain_loss,
                train_rmse=mtrain_rmse,
                train_mape=mtrain_mape,
                train_wmape=mtrain_wmape,
                valid_loss=mvalid_loss,
                valid_rmse=mvalid_rmse,
                valid_mape=mvalid_mape,
                valid_wmape=mvalid_wmape,
            )
        )
        result.append(train_m)

        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}".format(
                i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape
            ),
            flush=True,
        )
        print(
            "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}".format(
                i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape
            ),
            flush=True,
        )

        if mvalid_loss < best_valid_loss:
            print("###Update tasks appear###")
            best_valid_loss = mvalid_loss
            torch.save(engine.model.state_dict(), os.path.join(path, "best_model.pth"))
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

    horizon_metrics = evaluate_testset(engine, dataloader, scaler, device, args.output_len)

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
