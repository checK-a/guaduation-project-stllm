import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .layers import GraphConvLayer
from .utils import normalize_adj2, sparse_mx_to_torch_sparse_tensor


def _to_numpy_adj(orig_adj):
    if isinstance(orig_adj, np.ndarray):
        return orig_adj
    if torch.is_tensor(orig_adj):
        return orig_adj.detach().cpu().numpy()
    return np.asarray(orig_adj)


def _normalized_dense_adj(orig_adj, use_cuda):
    adj_np = _to_numpy_adj(orig_adj)
    adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(adj_np)).to_dense()
    if use_cuda and torch.cuda.is_available():
        return adj.cuda()
    return adj


class cola_gnn(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.x_h = 1
        self.f_h = data.m
        self.m = data.m
        self.d = data.d
        self.w = args.window
        self.h = args.horizon
        self.adj = data.adj
        self.o_adj = data.orig_adj
        self.adj = _normalized_dense_adj(data.orig_adj, args.cuda)
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        half_hid = int(self.n_hidden / 2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu
        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernel = self.w // 2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernel, dilation=2)
        long_out = self.w - 2 * (long_kernel - 1)
        self.n_spatial = 10
        self.conv1 = GraphConvLayer((1 + long_out) * self.k, self.n_hidden)
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)

        if args.rnn_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.x_h,
                hidden_size=self.n_hidden,
                num_layers=args.n_layer,
                dropout=args.dropout,
                batch_first=True,
                bidirectional=args.bi,
            )
        elif args.rnn_model == "GRU":
            self.rnn = nn.GRU(
                input_size=self.x_h,
                hidden_size=self.n_hidden,
                num_layers=args.n_layer,
                dropout=args.dropout,
                batch_first=True,
                bidirectional=args.bi,
            )
        elif args.rnn_model == "RNN":
            self.rnn = nn.RNN(
                input_size=self.x_h,
                hidden_size=self.n_hidden,
                num_layers=args.n_layer,
                dropout=args.dropout,
                batch_first=True,
                bidirectional=args.bi,
            )
        else:
            raise LookupError("only support LSTM, GRU and RNN")

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, self.h)

        self.residual_window = 0
        self.ratio = 1.0
        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        b, _, _ = x.size()
        orig_x = x
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        r_out, _ = self.rnn(x, None)
        last_hid = r_out[:, -1, :]
        last_hid = last_hid.view(-1, self.m, self.n_hidden)
        out_temporal = last_hid
        hid_rpt_m = last_hid.repeat(1, self.m, 1).view(b, self.m, self.m, self.n_hidden)
        hid_rpt_w = last_hid.repeat(1, 1, self.m).view(b, self.m, self.m, self.n_hidden)
        a_mx = self.act(hid_rpt_m @ self.W1.t() + hid_rpt_w @ self.W2.t() + self.b1) @ self.V + self.bv
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12)

        r_l = []
        r_long_l = []
        for i in range(self.m):
            h_tmp = orig_x[:, :, i : i + 1].permute(0, 2, 1).contiguous()
            r_l.append(self.conv(h_tmp))
            r_long_l.append(self.conv_long(h_tmp))
        r_l = torch.stack(r_l, dim=1)
        r_long_l = torch.stack(r_long_l, dim=1)
        r_l = torch.cat((r_l, r_long_l), -1)
        r_l = r_l.view(r_l.size(0), r_l.size(1), -1)
        r_l = torch.relu(r_l)

        adjs = self.adj.repeat(b, 1).view(b, self.m, self.m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        adj = adjs * c + a_mx * (1 - c)

        x = F.relu(self.conv1(r_l, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        out = torch.cat((out_spatial, out_temporal), dim=-1)
        out = self.out(out)
        out = out.permute(0, 2, 1).contiguous()

        if self.residual_window > 0:
            z = orig_x[:, -self.residual_window :, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1, self.m)
            out = out * self.ratio + z.unsqueeze(1)

        return out, None


class AR(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.m = data.m
        self.w = args.window
        self.h = args.horizon
        self.weight = Parameter(torch.Tensor(self.h, self.w, self.m))
        self.bias = Parameter(torch.zeros(self.h, self.m))
        nn.init.xavier_normal_(self.weight)

        args.output_fun = None
        self.output = None
        if args.output_fun == "sigmoid":
            self.output = F.sigmoid
        if args.output_fun == "tanh":
            self.output = F.tanh

    def forward(self, x):
        x = torch.einsum("bwm,hwm->bhm", x, self.weight) + self.bias.unsqueeze(0)
        if self.output is not None:
            x = self.output(x)
        return x, None


class VAR(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.m = data.m
        self.w = args.window
        self.h = args.horizon
        self.linear = nn.Linear(self.m * self.w, self.m * self.h)
        args.output_fun = None
        self.output = None
        if args.output_fun == "sigmoid":
            self.output = F.sigmoid
        if args.output_fun == "tanh":
            self.output = F.tanh

    def forward(self, x):
        x = x.view(-1, self.m * self.w)
        x = self.linear(x)
        x = x.view(-1, self.h, self.m)
        if self.output is not None:
            x = self.output(x)
        return x, None


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        temp = self.conv1(x) + torch.sigmoid(self.conv2(x))
        out = F.relu(temp + self.conv3(x))
        return out.permute(0, 2, 3, 1)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super().__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, x, a_hat):
        t = self.temporal1(x)
        lfs = torch.einsum("ij,jklm->kilm", [a_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class STGCN(nn.Module):
    def __init__(self, args, data, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super().__init__()
        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=args.n_hidden,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.block2 = STGCNBlock(
            in_channels=args.n_hidden,
            out_channels=args.n_hidden,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.last_temporal = TimeBlock(in_channels=args.n_hidden, out_channels=args.n_hidden)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * args.n_hidden, num_timesteps_output)
        self.adj = _normalized_dense_adj(data.orig_adj, args.cuda)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        out1 = self.block1(x, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4.permute(0, 2, 1).contiguous(), None
