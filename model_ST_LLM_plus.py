import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as Fuct
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model


class TemporalEmbedding(nn.Module):
    def __init__(self, features):
        super(TemporalEmbedding, self).__init__()
        self.week_embedding = nn.Embedding(54, features)
        self.day_of_week_embedding = nn.Embedding(7, features)
        self.day_of_year_embedding = nn.Embedding(366, features)
        nn.init.xavier_uniform_(self.week_embedding.weight)
        nn.init.xavier_uniform_(self.day_of_week_embedding.weight)
        nn.init.xavier_uniform_(self.day_of_year_embedding.weight)

    def forward(self, temporal_idx_x, num_nodes):
        if temporal_idx_x is None:
            raise ValueError("temporal_idx_x must not be None when using TemporalEmbedding.")

        if temporal_idx_x.ndim == 2:
            temporal_idx_x = temporal_idx_x.unsqueeze(-1)

        if temporal_idx_x.size(-1) == 1:
            current_week = temporal_idx_x[:, -1, 0].long().clamp(
                0, self.week_embedding.num_embeddings - 1
            )
            temporal_emb = self.week_embedding(current_week)
        else:
            current_dow = temporal_idx_x[:, -1, 0].long().clamp(
                0, self.day_of_week_embedding.num_embeddings - 1
            )
            current_doy = temporal_idx_x[:, -1, 1].long().clamp(
                0, self.day_of_year_embedding.num_embeddings - 1
            )
            temporal_emb = self.day_of_week_embedding(current_dow) + self.day_of_year_embedding(
                current_doy
            )

        return temporal_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_nodes, -1)


class TemporalSequenceEmbedding(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.week_embedding = nn.Embedding(54, features)
        self.day_of_week_embedding = nn.Embedding(7, features)
        self.day_of_year_embedding = nn.Embedding(366, features)
        nn.init.xavier_uniform_(self.week_embedding.weight)
        nn.init.xavier_uniform_(self.day_of_week_embedding.weight)
        nn.init.xavier_uniform_(self.day_of_year_embedding.weight)

    def forward(self, temporal_idx_x):
        if temporal_idx_x is None:
            raise ValueError("temporal_idx_x must not be None when using TemporalSequenceEmbedding.")

        if temporal_idx_x.ndim == 2:
            temporal_idx_x = temporal_idx_x.unsqueeze(-1)

        if temporal_idx_x.size(-1) == 1:
            week_idx = temporal_idx_x[..., 0].long().clamp(0, self.week_embedding.num_embeddings - 1)
            temporal_emb = self.week_embedding(week_idx)
        else:
            dow_idx = temporal_idx_x[..., 0].long().clamp(
                0, self.day_of_week_embedding.num_embeddings - 1
            )
            doy_idx = temporal_idx_x[..., 1].long().clamp(
                0, self.day_of_year_embedding.num_embeddings - 1
            )
            temporal_emb = self.day_of_week_embedding(dow_idx) + self.day_of_year_embedding(doy_idx)

        return temporal_emb


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1, dropout_rate=0.0):
        super(PFA, self).__init__()
        gpt2_path = "/root/gpt2_weights" if os.path.exists("/root/gpt2_weights") else "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(
            gpt2_path,
            attn_implementation="eager",
            output_attentions=True,
            output_hidden_states=True,
        )

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = 16

        unfrozen_layer_indices = list(range(gpt_layers - self.U, gpt_layers))
        lora_target_modules = [f"h.{i}.attn.c_attn" for i in unfrozen_layer_indices]
        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=lora_target_modules,
            bias="none",
        )
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

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

    def custom_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        adjacency_matrix: Optional[torch.FloatTensor] = None,
        graph_bias: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.gpt2.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.gpt2.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        total_layers = len(self.gpt2.h)
        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            layer_attention_mask = attention_mask.to(hidden_states.device) if attention_mask is not None else None
            if i >= total_layers - self.U:
                if adjacency_matrix is not None:
                    adjacency_bias = adjacency_matrix.to(hidden_states.device).float()
                    layer_attention_mask = (
                        adjacency_bias
                        if layer_attention_mask is None
                        else layer_attention_mask + adjacency_bias
                    )
                if graph_bias is not None:
                    graph_layer_bias = graph_bias.to(hidden_states.device).float()
                    layer_attention_mask = (
                        graph_layer_bias
                        if layer_attention_mask is None
                        else layer_attention_mask + graph_layer_bias
                    )

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=layer_attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)

        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward(self, x, adjacency_matrix=None, graph_bias=None):
        batch_size = x.shape[0]
        num_heads = self.gpt2.config.n_head
        expanded_adjacency = None
        expanded_graph_bias = None

        if adjacency_matrix is not None:
            if adjacency_matrix.dim() == 2:
                expanded_adjacency = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            elif adjacency_matrix.dim() == 3:
                if adjacency_matrix.size(0) != batch_size:
                    raise ValueError(
                        "Batch-wise adjacency must have the same batch size as the input: "
                        f"got {adjacency_matrix.size(0)} and {batch_size}."
                    )
                expanded_adjacency = adjacency_matrix
            else:
                raise ValueError(
                    f"adjacency_matrix must be [N, N] or [B, N, N], got {tuple(adjacency_matrix.shape)}."
                )
            expanded_adjacency = expanded_adjacency.unsqueeze(1).repeat(1, num_heads, 1, 1)
            expanded_adjacency = expanded_adjacency.to(self.device).float()

        if graph_bias is not None:
            if graph_bias.dim() == 2:
                expanded_graph_bias = graph_bias.unsqueeze(0).repeat(batch_size, 1, 1)
            elif graph_bias.dim() == 3:
                if graph_bias.size(0) != batch_size:
                    raise ValueError(
                        "Batch-wise graph_bias must have the same batch size as the input: "
                        f"got {graph_bias.size(0)} and {batch_size}."
                    )
                expanded_graph_bias = graph_bias
            else:
                raise ValueError(f"graph_bias must be [N, N] or [B, N, N], got {tuple(graph_bias.shape)}.")
            expanded_graph_bias = expanded_graph_bias.unsqueeze(1).repeat(1, num_heads, 1, 1)
            expanded_graph_bias = expanded_graph_bias.to(self.device).float()

        output = self.custom_forward(
            inputs_embeds=x,
            adjacency_matrix=expanded_adjacency,
            graph_bias=expanded_graph_bias,
        ).last_hidden_state
        output = self.dropout(output)
        return output


class EncoderBackboneMixin:
    gpt_channel = 256
    hidden_dim = 768

    def _init_encoder_backbone(self):
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, self.gpt_channel, kernel_size=(1, 1)
        )
        self.temporal_emb = TemporalEmbedding(self.gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)
        self.in_layer = nn.Conv2d(self.gpt_channel * 3, self.hidden_dim, kernel_size=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gpt = PFA(
            device=self.device,
            gpt_layers=self.llm_layer,
            U=self.U,
            dropout_rate=self.dropout_rate,
        )

    def encode_base(self, history_data, temporal_idx_x=None):
        data = history_data.permute(0, 3, 2, 1)
        batch_size, _, num_nodes, _ = data.shape

        if temporal_idx_x is None:
            temporal_idx_x = torch.zeros(
                (batch_size, self.input_len, 1), dtype=torch.long, device=history_data.device
            )

        time_emb = self.temporal_emb(temporal_idx_x, self.num_nodes)
        node_emb = (
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = data.permute(0, 3, 2, 1)
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        input_data = self.start_conv(input_data)

        data_st = torch.cat([input_data, time_emb, node_emb], dim=1)
        data_st = self.in_layer(data_st)
        data_st = Fuct.leaky_relu(data_st)
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        return data_st

    def encode(self, history_data, temporal_idx_x=None, use_llm=True, llm_fusion_mode=None):
        base_encoded = self.encode_base(history_data, temporal_idx_x)
        if not use_llm:
            return base_encoded

        fusion_mode = llm_fusion_mode or getattr(self, "llm_fusion_mode", "direct")
        if fusion_mode == "none":
            return base_encoded

        llm_encoded = self.gpt(base_encoded, self.adj_mx)
        if fusion_mode == "direct":
            return llm_encoded
        if fusion_mode == "residual_gate":
            gate = torch.sigmoid(self.llm_gate_head(base_encoded))
            return base_encoded + gate * (llm_encoded - base_encoded)

        raise ValueError(f"Unsupported llm_fusion_mode: {fusion_mode}")


class ST_LLM(nn.Module, EncoderBackboneMixin):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=3,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=6,
        U=1,
        use_llm=True,
    ):
        super().__init__()
        self.device = device
        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(self.device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.use_llm = use_llm
        self.dropout_rate = 0.1
        self.llm_fusion_mode = "direct"

        self._init_encoder_backbone()
        self.regression_layer = nn.Conv2d(self.hidden_dim, self.output_len, kernel_size=(1, 1))

    def param_num(self):
        return sum(param.nelement() for param in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, history_data, temporal_idx_x=None):
        encoded = self.encode(history_data, temporal_idx_x, use_llm=self.use_llm)
        outputs = encoded.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)
        return outputs


class GraphContextLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_states, adjacency):
        if adjacency.dim() == 2:
            aggregated = torch.einsum("nm,bmd->bnd", adjacency, node_states)
        elif adjacency.dim() == 3:
            aggregated = torch.bmm(adjacency, node_states)
        else:
            raise ValueError(f"Expected adjacency [N, N] or [B, N, N], got {tuple(adjacency.shape)}.")
        return self.norm(node_states + self.proj(aggregated))


class DynamicGraphHead(nn.Module):
    def __init__(self, hidden_dim, graph_dim=64, dropout_rate=0.1):
        super().__init__()
        context_dim = hidden_dim * 3
        self.query_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, graph_dim),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, graph_dim),
        )
        self.graph_dim = graph_dim

    def forward(self, base_states, graph_states):
        global_context = base_states.mean(dim=1, keepdim=True).expand(-1, base_states.size(1), -1)
        context = torch.cat([base_states, graph_states, global_context], dim=-1)
        query = self.query_proj(context)
        key = self.key_proj(context)
        logits = torch.bmm(query, key.transpose(1, 2)) / (self.graph_dim ** 0.5)
        return torch.sigmoid(logits)


class DynamicTransmissionSTLLM(nn.Module, EncoderBackboneMixin):
    def __init__(
        self,
        device,
        adj_mx,
        semantic_adj_mx=None,
        input_dim=1,
        num_nodes=51,
        input_len=24,
        output_len=4,
        llm_layer=6,
        U=1,
        use_llm=True,
        dt_graph_mode="static_semantic_dynamic",
        dynamic_graph_top_k=10,
        semantic_graph_top_k=8,
        dynamic_graph_alpha_init=1.0,
        dynamic_graph_dim=64,
    ):
        super().__init__()
        supported_modes = {"static", "dynamic", "static_dynamic", "static_semantic_dynamic"}
        if dt_graph_mode not in supported_modes:
            raise ValueError(f"Unsupported dt_graph_mode: {dt_graph_mode}")

        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.use_llm = use_llm
        self.dropout_rate = 0.1
        self.llm_fusion_mode = "direct"
        self.dt_graph_mode = dt_graph_mode
        self.dynamic_graph_top_k = int(dynamic_graph_top_k)
        self.semantic_graph_top_k = int(semantic_graph_top_k)

        static_adj = torch.tensor(adj_mx, dtype=torch.float32)
        if semantic_adj_mx is None:
            semantic_adj = torch.zeros_like(static_adj)
            semantic_adj.fill_diagonal_(1.0)
        else:
            semantic_adj = torch.tensor(semantic_adj_mx, dtype=torch.float32)

        self.register_buffer("adj_mx", static_adj.to(self.device), persistent=False)
        self.register_buffer("semantic_adj_mx", semantic_adj.to(self.device), persistent=False)
        # Start from a static-dominant blend; semantic edges can earn weight later.
        self.static_semantic_logits = nn.Parameter(torch.tensor([2.0, -2.0], dtype=torch.float32))
        self.dynamic_alpha_logit = nn.Parameter(torch.tensor(float(dynamic_graph_alpha_init)))

        self._init_encoder_backbone()
        self.graph_context = GraphContextLayer(self.hidden_dim, self.dropout_rate)
        self.dynamic_graph_head = DynamicGraphHead(
            self.hidden_dim,
            graph_dim=int(dynamic_graph_dim),
            dropout_rate=self.dropout_rate,
        )
        self.regression_layer = nn.Conv2d(self.hidden_dim, self.output_len, kernel_size=(1, 1))

    @staticmethod
    def _row_normalize(adjacency):
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _normalize_static(self, adjacency):
        adjacency = adjacency.float().clone()
        adjacency = torch.clamp(adjacency, min=0.0)
        adjacency.fill_diagonal_(1.0)
        return self._row_normalize(adjacency)

    def _build_base_graph(self):
        static_adj = self._normalize_static(self.adj_mx)
        if self.dt_graph_mode in {"static", "static_dynamic"}:
            return static_adj

        semantic_adj = self._normalize_static(self.semantic_adj_mx)
        weights = torch.softmax(self.static_semantic_logits, dim=0)
        base_graph = weights[0] * static_adj + weights[1] * semantic_adj
        return self._row_normalize(base_graph)

    def _sparsify_dynamic_graph(self, dynamic_graph):
        batch_size, num_nodes, _ = dynamic_graph.shape
        device = dynamic_graph.device
        top_k = max(0, min(int(self.dynamic_graph_top_k), num_nodes - 1))
        sparse = torch.zeros_like(dynamic_graph)
        off_diag = dynamic_graph.clone()
        eye = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)
        off_diag = off_diag.masked_fill(eye, -1.0)
        if top_k > 0:
            _, indices = torch.topk(off_diag, k=top_k, dim=-1)
            sparse.scatter_(-1, indices, dynamic_graph.gather(-1, indices))
        sparse = sparse.masked_fill(eye, 1.0)
        return self._row_normalize(sparse)

    @staticmethod
    def _graph_to_bias(graph):
        # Convert normalized graph weights into a stronger attention bias:
        # larger edges stay near 0, weak / pruned edges become strongly negative.
        return torch.log(graph.clamp_min(1e-6))

    def build_effective_graph(self, base_encoded):
        base_graph = self._build_base_graph().to(base_encoded.device)
        if self.dt_graph_mode == "static":
            return base_graph.unsqueeze(0).expand(base_encoded.size(0), -1, -1)

        graph_context = self.graph_context(base_encoded, base_graph)
        dynamic_graph = self._sparsify_dynamic_graph(
            self.dynamic_graph_head(base_encoded, graph_context)
        )

        if self.dt_graph_mode == "dynamic":
            effective_graph = dynamic_graph
        else:
            alpha = torch.sigmoid(self.dynamic_alpha_logit)
            base_graph_batch = base_graph.unsqueeze(0).expand(base_encoded.size(0), -1, -1)
            effective_graph = (1.0 - alpha) * base_graph_batch + alpha * dynamic_graph
            effective_graph = self._row_normalize(effective_graph)

        return effective_graph

    def encode(self, history_data, temporal_idx_x=None, use_llm=True, llm_fusion_mode=None):
        base_encoded = self.encode_base(history_data, temporal_idx_x)
        if not use_llm:
            return base_encoded

        effective_graph = self.build_effective_graph(base_encoded)
        graph_bias = self._graph_to_bias(effective_graph)
        return self.gpt(base_encoded, graph_bias=graph_bias)

    def forward(self, history_data, temporal_idx_x=None):
        encoded = self.encode(history_data, temporal_idx_x, use_llm=self.use_llm)
        outputs = encoded.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)
        return outputs


class EpiSTLLMPlus(nn.Module, EncoderBackboneMixin):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=1,
        num_nodes=51,
        input_len=24,
        output_len=10,
        llm_layer=6,
        U=1,
        compartment_dim=16,
        ablation_mode="full",
        llm_fusion_mode="direct",
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.dropout_rate = 0.1
        self.compartment_dim = compartment_dim
        self.global_context_dim = 256
        self.ablation_mode = ablation_mode
        self.llm_fusion_mode = llm_fusion_mode

        adj_tensor = torch.tensor(adj_mx, dtype=torch.float32)
        self.adj_mx = adj_tensor.to(self.device)
        self.register_buffer("adj_mx_norm", self._normalize_adjacency(adj_tensor), persistent=False)

        self._init_encoder_backbone()
        self.llm_gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim),
        )
        nn.init.constant_(self.llm_gate_head[-1].bias, -2.0)

        self.global_trend = nn.Sequential(
            nn.Linear(self.hidden_dim, self.global_context_dim),
            nn.GELU(),
            nn.Linear(self.global_context_dim, self.global_context_dim),
        )
        joint_dim = self.hidden_dim + self.global_context_dim
        self.beta_head = nn.Sequential(
            nn.Linear(joint_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len),
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(joint_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len),
        )

        self.s0_head = nn.Linear(self.hidden_dim, self.compartment_dim)
        self.i0_head = nn.Linear(self.hidden_dim, self.compartment_dim)
        self.r0_head = nn.Linear(self.hidden_dim, self.compartment_dim)
        self.i0_recent_scale = nn.Parameter(torch.tensor(1.0))

        infection_input_dim = self.compartment_dim * 2
        recovery_input_dim = self.compartment_dim
        self.infection_mlp = nn.Sequential(
            nn.Linear(infection_input_dim, self.compartment_dim),
            nn.GELU(),
            nn.Linear(self.compartment_dim, self.compartment_dim),
        )
        self.recovery_mlp = nn.Sequential(
            nn.Linear(recovery_input_dim, self.compartment_dim),
            nn.GELU(),
            nn.Linear(self.compartment_dim, self.compartment_dim),
        )
        self.observation_head = nn.Sequential(
            nn.Linear(self.compartment_dim * 2, self.compartment_dim),
            nn.GELU(),
            nn.Linear(self.compartment_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.output_len),
        )

        supported_modes = {"full", "no_mech", "mech_only", "no_llm", "fixed_params"}
        if self.ablation_mode not in supported_modes:
            raise ValueError(f"Unsupported ablation_mode: {self.ablation_mode}")
        supported_fusion_modes = {"direct", "none", "residual_gate"}
        if self.llm_fusion_mode not in supported_fusion_modes:
            raise ValueError(f"Unsupported llm_fusion_mode: {self.llm_fusion_mode}")

    @staticmethod
    def _normalize_adjacency(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.clone()
        degree = adjacency_matrix.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency_matrix / degree

    def load_encoder_state(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        own_state = self.state_dict()
        matched_state = {}
        for key, value in state_dict.items():
            if key in own_state and own_state[key].shape == value.shape:
                matched_state[key] = value

        missing, unexpected = self.load_state_dict(matched_state, strict=False)
        if not matched_state:
            raise RuntimeError(
                "No encoder weights were loaded from the warm-start checkpoint. "
                "Expected a checkpoint saved from st_llm_plus."
            )
        return missing, unexpected

    def freeze_encoder_for_stage2(self):
        encoder_modules = [
            self.start_conv,
            self.temporal_emb,
            self.in_layer,
            self.gpt,
        ]
        self.node_emb.requires_grad = False
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False

    def enable_joint_tuning_stage3(self):
        self.freeze_encoder_for_stage2()
        for name, param in self.gpt.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        for param in self.gpt.gpt2.h[-1].parameters():
            param.requires_grad = True

    def _compute_global_context(self, encoded):
        pooled = encoded.mean(dim=1)
        global_context = self.global_trend(pooled)
        return global_context.unsqueeze(1).expand(-1, self.num_nodes, -1)

    def _init_compartments(self, encoded, history_data):
        history_sequence = history_data.permute(0, 3, 2, 1)[..., 0]
        recent_window = min(3, history_sequence.size(1))
        recent_cases = history_sequence[:, -recent_window:, :].mean(dim=1, keepdim=False).unsqueeze(-1)
        recent_anchor = recent_cases.expand(-1, -1, self.compartment_dim)

        s0 = Fuct.softplus(self.s0_head(encoded))
        i0 = Fuct.softplus(self.i0_head(encoded) + self.i0_recent_scale * recent_anchor)
        r0 = Fuct.softplus(self.r0_head(encoded))
        return s0, i0, r0

    def _predict_parameters(self, joint_context):
        beta = torch.sigmoid(self.beta_head(joint_context)).permute(0, 2, 1).unsqueeze(-1)
        gamma = torch.sigmoid(self.gamma_head(joint_context)).permute(0, 2, 1).unsqueeze(-1)
        if self.ablation_mode == "fixed_params":
            beta = beta.mean(dim=1, keepdim=True).repeat(1, self.output_len, 1, 1)
            gamma = gamma.mean(dim=1, keepdim=True).repeat(1, self.output_len, 1, 1)
        return beta, gamma

    def _build_no_mech_outputs(self, residual, beta, gamma, s0, i0, r0):
        batch_size = residual.size(0)
        zero_prediction = torch.clamp_min(residual, 0.0)
        zero_mech = torch.zeros_like(residual)
        state_shape = (batch_size, self.output_len, self.num_nodes, self.compartment_dim)
        zero_states = torch.zeros(state_shape, device=residual.device, dtype=residual.dtype)
        flow_shape = (batch_size, self.output_len, self.num_nodes, 1)
        zero_flows = torch.zeros(flow_shape, device=residual.device, dtype=residual.dtype)
        return {
            "prediction": zero_prediction,
            "beta": beta,
            "gamma": gamma,
            "s0": s0,
            "i0": i0,
            "r0": r0,
            "S": zero_states,
            "I": zero_states,
            "R": zero_states,
            "delta_inf": zero_flows,
            "delta_rec": zero_flows,
            "y_mech": zero_mech,
            "y_res": residual,
            "skip_mech_regularizers": True,
        }

    def _rollout(self, beta, gamma, s0, i0, r0):
        s_prev, i_prev, r_prev = s0, i0, r0
        s_states = []
        i_states = []
        r_states = []
        mech_outputs = []
        inf_flows = []
        rec_flows = []

        for horizon_index in range(self.output_len):
            beta_t = beta[:, horizon_index]
            gamma_t = gamma[:, horizon_index]
            lambda_t = torch.einsum("nm,bmd->bnd", self.adj_mx_norm, i_prev)

            inf_base = Fuct.softplus(self.infection_mlp(torch.cat([s_prev, lambda_t], dim=-1)))
            rec_base = Fuct.softplus(self.recovery_mlp(i_prev))
            delta_inf = torch.minimum(beta_t * inf_base, s_prev)
            delta_rec = torch.minimum(gamma_t * rec_base, i_prev + delta_inf)

            y_mech_t = Fuct.softplus(
                self.observation_head(torch.cat([delta_inf, i_prev], dim=-1))
            )

            s_prev = torch.clamp_min(s_prev - delta_inf, 0.0)
            i_prev = torch.clamp_min(i_prev + delta_inf - delta_rec, 0.0)
            r_prev = torch.clamp_min(r_prev + delta_rec, 0.0)

            s_states.append(s_prev)
            i_states.append(i_prev)
            r_states.append(r_prev)
            mech_outputs.append(y_mech_t)
            inf_flows.append(delta_inf)
            rec_flows.append(delta_rec)

        return (
            torch.stack(mech_outputs, dim=1),
            torch.stack(s_states, dim=1),
            torch.stack(i_states, dim=1),
            torch.stack(r_states, dim=1),
            torch.stack(inf_flows, dim=1),
            torch.stack(rec_flows, dim=1),
        )

    def forward(self, history_data, temporal_idx_x=None, return_aux=False):
        encoded = self.encode(
            history_data,
            temporal_idx_x,
            use_llm=self.ablation_mode != "no_llm",
            llm_fusion_mode=self.llm_fusion_mode,
        )
        global_context = self._compute_global_context(encoded)
        joint_context = torch.cat([encoded, global_context], dim=-1)

        beta, gamma = self._predict_parameters(joint_context)
        s0, i0, r0 = self._init_compartments(encoded, history_data)
        residual = self.residual_head(encoded).permute(0, 2, 1).unsqueeze(-1)

        if self.ablation_mode == "no_mech":
            outputs = self._build_no_mech_outputs(residual, beta, gamma, s0, i0, r0)
            if not return_aux:
                return outputs["prediction"]
            return outputs

        y_mech, s_states, i_states, r_states, delta_inf, delta_rec = self._rollout(
            beta, gamma, s0, i0, r0
        )
        if self.ablation_mode == "mech_only":
            prediction = y_mech
        else:
            prediction = torch.clamp_min(y_mech + residual, 0.0)

        if not return_aux:
            return prediction

        return {
            "prediction": prediction,
            "beta": beta,
            "gamma": gamma,
            "s0": s0,
            "i0": i0,
            "r0": r0,
            "S": s_states,
            "I": i_states,
            "R": r_states,
            "delta_inf": delta_inf,
            "delta_rec": delta_rec,
            "y_mech": y_mech,
            "y_res": residual,
            "skip_mech_regularizers": False,
        }


class EpiSTLLMPlusV2b(EpiSTLLMPlus):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=1,
        num_nodes=51,
        input_len=24,
        output_len=10,
        llm_layer=6,
        U=1,
        compartment_dim=16,
        ablation_mode="full",
        llm_fusion_mode="residual_gate",
        temporal_patch_len=4,
        temporal_patch_stride=4,
        graph_bias_mode="patch_graph_bias",
        graph_bias_scale_init=1.0,
    ):
        self.temporal_patch_len = temporal_patch_len
        self.temporal_patch_stride = temporal_patch_stride
        self.graph_bias_mode = graph_bias_mode
        self.graph_bias_scale_init = graph_bias_scale_init
        if self.temporal_patch_stride != self.temporal_patch_len:
            raise ValueError("EpiSTLLMPlusV2b currently requires temporal_patch_stride == temporal_patch_len.")
        if input_len % temporal_patch_len != 0:
            raise ValueError("EpiSTLLMPlusV2b requires input_len % temporal_patch_len == 0.")
        self.patch_count = input_len // temporal_patch_len

        super().__init__(
            device,
            adj_mx,
            input_dim,
            num_nodes,
            input_len,
            output_len,
            llm_layer,
            U,
            compartment_dim,
            ablation_mode,
            llm_fusion_mode,
        )

        supported_graph_bias_modes = {"patch_graph_bias", "none"}
        if self.graph_bias_mode not in supported_graph_bias_modes:
            raise ValueError(f"Unsupported graph_bias_mode: {self.graph_bias_mode}")

        graph_adj = self._normalize_adjacency(
            self.adj_mx.detach().clone() + torch.eye(self.num_nodes, device=self.adj_mx.device)
        )
        patch_graph_bias = torch.kron(
            torch.eye(self.patch_count, device=graph_adj.device, dtype=graph_adj.dtype),
            graph_adj,
        )
        self.register_buffer("patch_graph_bias", patch_graph_bias, persistent=False)
        self.graph_bias_scale = nn.Parameter(
            torch.tensor(float(self.graph_bias_scale_init), device=self.adj_mx.device)
        )

    def _init_encoder_backbone(self):
        self.temporal_seq_emb = TemporalSequenceEmbedding(self.gpt_channel)
        self.value_patch_conv = nn.Conv1d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=self.temporal_patch_len,
            stride=self.temporal_patch_stride,
        )
        self.time_patch_conv = nn.Conv1d(
            self.gpt_channel,
            self.hidden_dim,
            kernel_size=self.temporal_patch_len,
            stride=self.temporal_patch_stride,
        )
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.hidden_dim))
        self.patch_pos_emb = nn.Parameter(torch.empty(self.patch_count, self.hidden_dim))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.patch_pos_emb)

        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gpt = PFA(
            device=self.device,
            gpt_layers=self.llm_layer,
            U=self.U,
            dropout_rate=self.dropout_rate,
        )
        self.readout_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def freeze_encoder_for_stage2(self):
        encoder_modules = [
            self.temporal_seq_emb,
            self.value_patch_conv,
            self.time_patch_conv,
            self.token_norm,
            self.readout_proj,
            self.gpt,
        ]
        self.node_emb.requires_grad = False
        self.patch_pos_emb.requires_grad = False
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False

    def _build_temporal_tokens(self, history_data, temporal_idx_x=None):
        history_sequence = history_data.permute(0, 3, 2, 1)
        batch_size, input_len, num_nodes, _ = history_sequence.shape
        if input_len != self.input_len:
            raise ValueError(
                f"EpiSTLLMPlusV2b expected history length {self.input_len}, got {input_len}."
            )

        if temporal_idx_x is None:
            temporal_idx_x = torch.zeros(
                (batch_size, self.input_len, 1), dtype=torch.long, device=history_data.device
            )

        value_series = history_sequence.permute(0, 2, 3, 1).reshape(
            batch_size * num_nodes, self.input_dim, self.input_len
        )
        value_tokens = self.value_patch_conv(value_series)
        value_tokens = value_tokens.transpose(1, 2).reshape(
            batch_size, num_nodes, self.patch_count, self.hidden_dim
        )
        value_tokens = value_tokens.permute(0, 2, 1, 3)

        time_sequence_emb = self.temporal_seq_emb(temporal_idx_x).transpose(1, 2)
        time_tokens = self.time_patch_conv(time_sequence_emb).transpose(1, 2).unsqueeze(2)
        time_tokens = time_tokens.expand(-1, -1, num_nodes, -1)

        node_tokens = self.node_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, self.patch_count, -1, -1)
        patch_tokens = self.patch_pos_emb.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_nodes, -1)

        token_base = value_tokens + time_tokens + node_tokens + patch_tokens
        token_base = self.token_norm(token_base)
        token_base = self.dropout(token_base)
        return token_base

    def _readout_tokens(self, token_tensor):
        last_patch_repr = token_tensor[:, -1, :, :]
        mean_patch_repr = token_tensor.mean(dim=1)
        return self.readout_proj(torch.cat([last_patch_repr, mean_patch_repr], dim=-1))

    def encode(self, history_data, temporal_idx_x=None, use_llm=True, llm_fusion_mode=None):
        token_base = self._build_temporal_tokens(history_data, temporal_idx_x)
        batch_size = token_base.size(0)
        token_base_flat = token_base.reshape(batch_size, self.patch_count * self.num_nodes, self.hidden_dim)

        if not use_llm:
            return self._readout_tokens(token_base)

        fusion_mode = llm_fusion_mode or getattr(self, "llm_fusion_mode", "residual_gate")
        if fusion_mode == "none":
            return self._readout_tokens(token_base)

        graph_bias = None
        if self.graph_bias_mode == "patch_graph_bias":
            graph_bias = self.graph_bias_scale * self.patch_graph_bias

        token_ctx_flat = self.gpt(token_base_flat, graph_bias=graph_bias)
        if fusion_mode == "direct":
            token_fused_flat = token_ctx_flat
        elif fusion_mode == "residual_gate":
            gate = torch.sigmoid(self.llm_gate_head(token_base_flat))
            token_fused_flat = token_base_flat + gate * (token_ctx_flat - token_base_flat)
        else:
            raise ValueError(f"Unsupported llm_fusion_mode: {fusion_mode}")

        token_fused = token_fused_flat.reshape(batch_size, self.patch_count, self.num_nodes, self.hidden_dim)
        return self._readout_tokens(token_fused)
