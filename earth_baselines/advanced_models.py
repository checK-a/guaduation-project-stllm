import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dense_row_normalized_adj(adj):
    if not torch.is_tensor(adj):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj = adj.float()
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    adj = torch.clamp(adj, min=0.0) + eye
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class Persistence(nn.Module):
    """Last-value baseline in the normalized data space."""

    def __init__(self, args, data):
        super().__init__()
        self.horizon = args.horizon

    def forward(self, x):
        last_value = x[:, -1:, :]
        return last_value.repeat(1, self.horizon, 1), None


class RecurrentBaseline(nn.Module):
    def __init__(self, args, data, cell_type="GRU"):
        super().__init__()
        self.num_nodes = data.m
        self.input_len = args.window
        self.horizon = args.horizon
        self.hidden_dim = args.n_hidden
        self.cell_type = cell_type.upper()
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[self.cell_type]
        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=args.n_layer,
            dropout=args.dropout if args.n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(args.dropout)
        self.proj = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x):
        batch_size, _, num_nodes = x.shape
        node_series = x.permute(0, 2, 1).contiguous().view(batch_size * num_nodes, -1, 1)
        output, _ = self.rnn(node_series)
        last_hidden = self.dropout(output[:, -1])
        pred = self.proj(last_hidden)
        pred = pred.view(batch_size, num_nodes, self.horizon).permute(0, 2, 1).contiguous()
        return pred, None


class GRUBaseline(RecurrentBaseline):
    def __init__(self, args, data):
        super().__init__(args, data, cell_type="GRU")


class LSTMBaseline(RecurrentBaseline):
    def __init__(self, args, data):
        super().__init__(args, data, cell_type="LSTM")


class DiffusionGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, diffusion_steps=2):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.proj = nn.Linear(input_dim * (2 * diffusion_steps + 1), output_dim)

    def forward(self, x, adj):
        supports = [x]
        forward_x = x
        backward_x = x
        adj_t = adj.transpose(0, 1)
        for _ in range(self.diffusion_steps):
            forward_x = torch.einsum("nm,bmd->bnd", adj, forward_x)
            backward_x = torch.einsum("nm,bmd->bnd", adj_t, backward_x)
            supports.extend([forward_x, backward_x])
        return self.proj(torch.cat(supports, dim=-1))


class DCRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, diffusion_steps=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = DiffusionGraphConv(input_dim + hidden_dim, 2 * hidden_dim, diffusion_steps)
        self.update_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, diffusion_steps)

    def forward(self, x_t, h_prev, adj):
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        reset_gate, update_gate = torch.sigmoid(self.gate_conv(gate_input, adj)).chunk(2, dim=-1)
        candidate_input = torch.cat([x_t, reset_gate * h_prev], dim=-1)
        candidate = torch.tanh(self.update_conv(candidate_input, adj))
        return update_gate * h_prev + (1.0 - update_gate) * candidate


class DCRNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.num_nodes = data.m
        self.horizon = args.horizon
        self.hidden_dim = args.n_hidden
        self.register_buffer("adj", _dense_row_normalized_adj(data.orig_adj), persistent=False)
        self.cell = DCRNNCell(1, self.hidden_dim, diffusion_steps=2)
        self.dropout = nn.Dropout(args.dropout)
        self.proj = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x):
        batch_size, input_len, num_nodes = x.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        adj = self.adj.to(x.device, x.dtype)
        for t in range(input_len):
            h = self.cell(x[:, t, :].unsqueeze(-1), h, adj)
        pred = self.proj(self.dropout(h)).permute(0, 2, 1).contiguous()
        return pred, None


class EpiGNNLite(nn.Module):
    """Compact epidemic-aware graph baseline.

    It combines per-node temporal encoding, a sample-adaptive similarity graph, and
    static adjacency gated graph propagation.
    """

    def __init__(self, args, data):
        super().__init__()
        self.num_nodes = data.m
        self.horizon = args.horizon
        self.hidden_dim = args.n_hidden
        self.register_buffer("static_adj", _dense_row_normalized_adj(data.orig_adj), persistent=False)
        self.temporal_encoder = nn.GRU(1, self.hidden_dim, batch_first=True)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.graph_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.graph_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.out = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x):
        batch_size, _, num_nodes = x.shape
        node_series = x.permute(0, 2, 1).reshape(batch_size * num_nodes, -1, 1)
        _, h = self.temporal_encoder(node_series)
        h = h[-1].view(batch_size, num_nodes, self.hidden_dim)

        q = self.query_proj(h)
        k = self.key_proj(h)
        adaptive_adj = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim), dim=-1)
        static_context = torch.einsum("nm,bmd->bnd", self.static_adj.to(x.device, x.dtype), h)
        dynamic_context = torch.bmm(adaptive_adj, h)
        gate = torch.sigmoid(self.graph_gate(torch.cat([h, static_context], dim=-1)))
        graph_context = gate * static_context + (1.0 - gate) * dynamic_context
        h = h + self.graph_update(torch.cat([h, graph_context], dim=-1))
        pred = self.out(h).permute(0, 2, 1).contiguous()
        return pred, None


class SIRBaseline(nn.Module):
    """Trainable SIR-inspired baseline that outputs original-scale predictions."""

    def __init__(self, args, data):
        super().__init__()
        self.num_nodes = data.m
        self.horizon = args.horizon
        self.register_buffer("adj", _dense_row_normalized_adj(data.orig_adj), persistent=False)
        self.register_buffer("scaler_mean", torch.tensor(float(getattr(args, "scaler_mean", 0.0))))
        self.register_buffer("scaler_std", torch.tensor(float(getattr(args, "scaler_std", 1.0))))
        self.beta_logit = nn.Parameter(torch.zeros(self.num_nodes))
        self.gamma_logit = nn.Parameter(torch.full((self.num_nodes,), -1.0))
        self.capacity_log = nn.Parameter(torch.zeros(self.num_nodes))
        self.report_log = nn.Parameter(torch.zeros(self.num_nodes))
        self.trend = nn.Linear(2, self.horizon)

    def forward(self, x):
        x_raw = torch.clamp_min(x * self.scaler_std.to(x.device) + self.scaler_mean.to(x.device), 0.0)
        i_prev = x_raw[:, -1, :]
        recent_mean = x_raw[:, -min(4, x_raw.size(1)) :, :].mean(dim=1)
        recent_slope = x_raw[:, -1, :] - x_raw[:, -min(4, x_raw.size(1)), :]
        trend_res = self.trend(torch.stack([recent_mean, recent_slope], dim=-1)).permute(0, 2, 1)

        beta = torch.sigmoid(self.beta_logit).view(1, -1)
        gamma = torch.sigmoid(self.gamma_logit).view(1, -1)
        capacity = F.softplus(self.capacity_log).view(1, -1) * (x_raw.amax(dim=1) + 1.0)
        susceptible = capacity
        report = F.softplus(self.report_log).view(1, -1)
        adj = self.adj.to(x.device, x.dtype)

        preds = []
        for _ in range(self.horizon):
            neighbor_i = torch.einsum("nm,bm->bn", adj, i_prev)
            new_inf = torch.minimum(beta * susceptible * neighbor_i / capacity.clamp_min(1e-6), susceptible)
            rec = torch.minimum(gamma * i_prev, i_prev + new_inf)
            susceptible = torch.clamp_min(susceptible - new_inf, 0.0)
            i_prev = torch.clamp_min(i_prev + new_inf - rec, 0.0)
            preds.append(report * new_inf + 0.2 * i_prev)

        sir_pred = torch.stack(preds, dim=1)
        return torch.clamp_min(sir_pred + trend_res, 0.0), None


class PatchTST(nn.Module):
    """Small PatchTST-style baseline with shared node-wise patch encoder."""

    def __init__(self, args, data):
        super().__init__()
        self.num_nodes = data.m
        self.input_len = args.window
        self.horizon = args.horizon
        self.hidden_dim = args.n_hidden
        self.patch_len = min(8, self.input_len)
        self.stride = max(1, self.patch_len // 2)
        self.num_patches = 1 + max(0, (self.input_len - self.patch_len) // self.stride)
        self.patch_proj = nn.Linear(self.patch_len, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4 if self.hidden_dim % 4 == 0 else 2,
            dim_feedforward=self.hidden_dim * 4,
            dropout=args.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, args.n_layer))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x):
        batch_size, _, num_nodes = x.shape
        node_series = x.permute(0, 2, 1).contiguous().view(batch_size * num_nodes, 1, self.input_len)
        patches = node_series.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        tokens = self.patch_proj(patches) + self.pos_emb[:, : patches.size(1)]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        pred = self.head(pooled).view(batch_size, num_nodes, self.horizon)
        return pred.permute(0, 2, 1).contiguous(), None
