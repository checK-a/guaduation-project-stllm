import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvLayer, GraphLearner, RegionAwareConv, get_laplace_mat


def _as_tensor_adj(adj):
    if torch.is_tensor(adj):
        return adj.detach().float()
    return torch.tensor(adj, dtype=torch.float32)


class EpiGNNModel(nn.Module):
    """EpiGNN adapted from the authors' released PyTorch code.

    The original code predicts one fixed horizon. This adapter keeps its
    transmission-risk encoding and region-aware graph learner, but changes the
    prediction head to output the full horizon [B, H, N] used by this project.
    """

    def __init__(self, args, data):
        super().__init__()
        self.num_nodes = data.m
        self.input_len = args.window
        self.horizon = args.horizon
        self.dropout_rate = args.dropout
        self.k = int(getattr(args, "epignn_k", args.k))
        self.hidA = int(getattr(args, "epignn_hidA", args.n_hidden))
        self.hidP = int(getattr(args, "epignn_hidP", 1))
        self.gcn_layers = int(getattr(args, "epignn_gcn_layers", 2))
        self.highway_window = int(getattr(args, "epignn_highway_window", 0))
        self.residual_concat = bool(int(getattr(args, "epignn_residual_concat", 0)))
        self.dropout = nn.Dropout(self.dropout_rate)

        adj = _as_tensor_adj(data.orig_adj)
        self.register_buffer("adj", adj, persistent=False)
        self.register_buffer("degree", adj.sum(dim=-1), persistent=False)

        # Official RegionAwareConv produces k * 4 * hidP + k channels.
        self.hidR = self.k * 4 * self.hidP + self.k
        self.backbone = RegionAwareConv(
            input_len=self.input_len,
            num_nodes=self.num_nodes,
            k=self.k,
            hidP=self.hidP,
        )

        # Global transmission risk encoding.
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)

        # Local transmission risk encoding from region degree.
        self.s_enc = nn.Linear(1, self.hidR)

        # Region-aware graph learner and static geographic gate.
        self.d_gate = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList(
            [GraphConvLayer(self.hidR, self.hidR) for _ in range(self.gcn_layers)]
        )

        if self.residual_concat:
            output_dim = self.hidR * (self.gcn_layers + 1)
        else:
            output_dim = self.hidR * 2
        self.output = nn.Linear(output_dim, self.horizon)

        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                stdv = 1.0 / math.sqrt(param.size(0))
                param.data.uniform_(-stdv, stdv)

    def _global_risk_encoding(self, temp_emb):
        query = self.dropout(self.WQ(temp_emb))
        key = self.dropout(self.WK(temp_emb))
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = attn.sum(dim=-1, keepdim=True)
        return self.dropout(self.t_enc(attn)), attn

    def _local_risk_encoding(self, device, dtype):
        degree = self.degree.to(device=device, dtype=dtype).unsqueeze(1)
        return self.dropout(self.s_enc(degree)), degree

    def _region_aware_graph(self, temp_emb, degree, batch_size):
        d_mat = torch.mm(degree, degree.transpose(1, 0))
        d_mat = torch.sigmoid(self.d_gate.to(degree.device, degree.dtype) * d_mat)
        spatial_adj = d_mat * self.adj.to(degree.device, degree.dtype)
        learned_adj = self.graphGen(temp_emb)
        return learned_adj + spatial_adj.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x, index=None, isEval=False):
        # x: [B, T, N]
        batch_size = x.shape[0]
        temp_emb = self.backbone(x)

        t_enc, attn = self._global_risk_encoding(temp_emb)
        s_enc, degree = self._local_risk_encoding(x.device, x.dtype)
        feat_emb = temp_emb + t_enc + s_enc.unsqueeze(0)

        adj = self._region_aware_graph(temp_emb, degree, batch_size)
        laplace_adj = get_laplace_mat(batch_size, self.num_nodes, adj)

        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = self.dropout(layer(node_state, laplace_adj))
            node_state_list.append(node_state)

        if self.residual_concat:
            node_repr = torch.cat(node_state_list + [feat_emb], dim=-1)
        else:
            node_repr = torch.cat([node_state, feat_emb], dim=-1)

        pred = self.output(node_repr).permute(0, 2, 1).contiguous()

        if self.highway_window > 0:
            z = x[:, -self.highway_window :, :].permute(0, 2, 1).contiguous()
            pred = pred + self.highway(z).permute(0, 2, 1).contiguous()

        imd = (adj, attn) if isEval else None
        return pred, imd
