import os

import torch
import torch.nn as nn
import torch_geometric

from transformers import GPT2Model
from torch_geometric.nn import GATConv

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        gpt2_path = "/root/gpt2_weights" if os.path.exists("/root/gpt2_weights") else "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(
            gpt2_path, output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class GATGPT(nn.Module):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.adj_mx = adj_mx
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.U = 2

        gpt_channel = 768
            
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.gat = GATConv(in_channels=gpt_channel, out_channels=gpt_channel)
        adj_tensor = torch.as_tensor(self.adj_mx, dtype=torch.float32)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj_tensor)
        self.register_buffer("edge_index", edge_index, persistent=False)

        # regression
        self.regression_layer = nn.Conv2d(gpt_channel, self.output_len, kernel_size=(1, 1))

        self.gpt = PFA(device=self.device, gpt_layers=6, U=self.U)
                 
    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data, temporal_idx_x=None):

        data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = data.shape

        input_data = data.transpose(1, 2).contiguous()
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))

        data_st = self.start_conv(input_data)

        # Reshape data for GNN
        data_flat = data_st.view(B * S, -1)

        data_st = self.gat(data_flat, self.edge_index) + data_flat
        data_st = data_st.view(B, S, -1)
        outputs = self.gpt(data_st)
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)

        # regression
        outputs = self.regression_layer(outputs)  

        return outputs
