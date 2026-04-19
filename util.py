import numpy as np
import os
import scipy.sparse as sp
import torch
import pickle
import json
from pathlib import Path

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class DataLoader(object):
    def __init__(
        self,
        xs,
        ys,
        batch_size,
        temporal_idx_xs=None,
        temporal_idx_ys=None,
        pad_with_last_sample=True,
    ):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            if temporal_idx_xs is not None:
                temporal_idx_x_padding = np.repeat(temporal_idx_xs[-1:], num_padding, axis=0)
                temporal_idx_xs = np.concatenate([temporal_idx_xs, temporal_idx_x_padding], axis=0)
            if temporal_idx_ys is not None:
                temporal_idx_y_padding = np.repeat(temporal_idx_ys[-1:], num_padding, axis=0)
                temporal_idx_ys = np.concatenate([temporal_idx_ys, temporal_idx_y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.temporal_idx_xs = temporal_idx_xs
        self.temporal_idx_ys = temporal_idx_ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        if self.temporal_idx_xs is not None:
            self.temporal_idx_xs = self.temporal_idx_xs[permutation]
        if self.temporal_idx_ys is not None:
            self.temporal_idx_ys = self.temporal_idx_ys[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                temporal_idx_x_i = None
                temporal_idx_y_i = None
                if self.temporal_idx_xs is not None:
                    temporal_idx_x_i = self.temporal_idx_xs[start_ind:end_ind, ...]
                if self.temporal_idx_ys is not None:
                    temporal_idx_y_i = self.temporal_idx_ys[start_ind:end_ind, ...]
                yield (x_i, y_i, temporal_idx_x_i, temporal_idx_y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    dataset_path = Path(dataset_dir)
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
        temporal_idx_x = None
        temporal_idx_y = None
        if "temporal_idx_x" in cat_data:
            temporal_idx_x = cat_data["temporal_idx_x"]
        elif "week_idx_x" in cat_data:
            temporal_idx_x = cat_data["week_idx_x"][..., None]
        elif "dow_idx_x" in cat_data and "doy_idx_x" in cat_data:
            temporal_idx_x = np.stack([cat_data["dow_idx_x"], cat_data["doy_idx_x"]], axis=-1)

        if "temporal_idx_y" in cat_data:
            temporal_idx_y = cat_data["temporal_idx_y"]
        elif "week_idx_y" in cat_data:
            temporal_idx_y = cat_data["week_idx_y"][..., None]
        elif "dow_idx_y" in cat_data and "doy_idx_y" in cat_data:
            temporal_idx_y = np.stack([cat_data["dow_idx_y"], cat_data["doy_idx_y"]], axis=-1)

        if temporal_idx_x is not None:
            data["temporal_idx_x_" + category] = temporal_idx_x.astype(np.int64)
        if temporal_idx_y is not None:
            data["temporal_idx_y_" + category] = temporal_idx_y.astype(np.int64)

    meta_path = dataset_path / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mean = meta["scaler_mean"]
        std = meta["scaler_std"]
        data["temporal_feature_names"] = meta.get("temporal_feature_names", [])
    else:
        mean = data["x_train"][..., 0].mean()
        std = data["x_train"][..., 0].std()
    if std == 0:
        std = 1.0
    scaler = StandardScaler(mean=mean, std=std)
    # Data format
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]
    if "temporal_idx_x_train" in data:
        data["temporal_idx_x_train"] = data["temporal_idx_x_train"][random_train, ...]
    if "temporal_idx_y_train" in data:
        data["temporal_idx_y_train"] = data["temporal_idx_y_train"][random_train, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(
        data["x_train"],
        data["y_train"],
        batch_size,
        temporal_idx_xs=data.get("temporal_idx_x_train"),
        temporal_idx_ys=data.get("temporal_idx_y_train"),
    )
    data["val_loader"] = DataLoader(
        data["x_val"],
        data["y_val"],
        valid_batch_size,
        temporal_idx_xs=data.get("temporal_idx_x_val"),
        temporal_idx_ys=data.get("temporal_idx_y_val"),
    )
    data["test_loader"] = DataLoader(
        data["x_test"],
        data["y_test"],
        test_batch_size,
        temporal_idx_xs=data.get("temporal_idx_x_test"),
        temporal_idx_ys=data.get("temporal_idx_y_test"),
    )
    data["scaler"] = scaler

    return data

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real,0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape
