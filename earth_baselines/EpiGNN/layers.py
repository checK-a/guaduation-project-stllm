import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.act = nn.ELU()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
            stdv = 1.0 / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = tanhalpha

    def forward(self, embedding):
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2)) - torch.bmm(
            nodevec2, nodevec1.transpose(1, 2)
        )
        return torch.relu(torch.tanh(self.alpha * adj))


class ConvBranch(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, kernel_size, dilation_factor, hidP=1, is_pool=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_pool = is_pool
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation_factor, 1),
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.is_pool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, num_nodes))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.batchnorm(self.conv(x))
        if self.is_pool:
            x = self.pooling(x)
        return x.view(batch_size, -1, self.num_nodes)


class RegionAwareConv(nn.Module):
    def __init__(self, input_len, num_nodes, k, hidP, dilation_factor=2):
        super().__init__()
        self.input_len = input_len
        self.num_nodes = num_nodes
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(num_nodes, 1, k, kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(num_nodes, 1, k, kernel_size=5, dilation_factor=1, hidP=hidP)
        self.conv_p1 = ConvBranch(num_nodes, 1, k, kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(num_nodes, 1, k, kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_g = ConvBranch(
            num_nodes,
            1,
            k,
            kernel_size=input_len,
            dilation_factor=1,
            hidP=None,
            is_pool=False,
        )
        self.activate = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 1, self.input_len, self.num_nodes)
        x_local = torch.cat([self.conv_l1(x), self.conv_l2(x)], dim=1)
        x_period = torch.cat([self.conv_p1(x), self.conv_p2(x)], dim=1)
        x_global = self.conv_g(x)
        return self.activate(torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1))


def get_laplace_mat(batch_size, num_nodes, adj):
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    ones = torch.ones(num_nodes, num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    eye = eye.expand(batch_size, num_nodes, num_nodes)
    ones = ones.expand(batch_size, num_nodes, num_nodes)
    adj = torch.where(adj > 0, ones, adj)
    degree = adj.sum(dim=2).unsqueeze(2).clamp_min(1e-12)
    degree_inv = torch.pow(degree, -1).expand(-1, -1, num_nodes)
    degree_mat = eye * degree_inv
    return torch.bmm(degree_mat, adj)
