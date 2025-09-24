import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


# -------- create activation and norm --------
def create_activation(name):
    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name, n, h=16):
    if name is None:
        return nn.Identity()
    elif name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name.startswith("groupnorm"):
        inferred_num_groups = int(name.repalce("groupnorm", ""))
        return nn.GroupNorm(inferred_num_groups, n)
    else:
        raise NotImplementedError(f"{name} is not implemented.")


# -------- projector and guidance layers --------
class EdgeRegressionHead(torch.nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, dropout=0., activate_func=F.relu, agg='concat', 
                 stop_grad=False, return_prob=False, **kwargs):
        super().__init__()
        self.stop_grad = stop_grad
        self.return_prob = return_prob

        self.agg = agg
        if self.agg == 'concat':
            in_channels = 2 * input_dim
        elif self.agg in ['mul', 'sum']:
            in_channels = input_dim
        else:
            raise NotImplementedError(f'Unsupported aggregation {self.agg}')

        self.mlp = MLP(num_layers=num_layers, in_channels=in_channels, hidden_channels=input_dim, out_channels=output_dim,
                       dropout=dropout, activate_func=activate_func)
    
    def forward(self, x, src, dst):
        if self.stop_grad:
            x = x.detach()

        if self.agg == 'concat':
            edge_emb = torch.cat([x[src], x[dst]], -1)
        elif self.agg == 'mul':
            edge_emb = x[src] * x[dst]
        elif self.agg == 'sum':
            edge_emb = x[src] + x[dst]
        else:
            raise NotImplementedError(f'Unsupported aggregation {self.agg}')

        edge_logits = self.mlp(edge_emb)
        if self.return_prob:
            edge_logits = torch.sigmoid(edge_logits)

        return edge_logits


class NodeRegressionHead(torch.nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim=None, dropout=0., activate_func=F.relu, 
                 stop_grad=False, **kwargs):
        super().__init__()
        self.stop_grad = stop_grad
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        self.regression_head = MLP(
            num_layers=num_layers, in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim,
            dropout=dropout, activate_func=activate_func
        )

    def forward(self, x):
        if self.stop_grad:
            x = x.detach()

        x = self.regression_head(x)
        return x


class GraphRegressionHead(torch.nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim=None, dropout=0., activate_func=F.relu, 
                 stop_grad=False, aggregation='mean', **kwargs):
        super().__init__()
        self.stop_grad = stop_grad
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        if aggregation == "sum":
            self.pooling = global_add_pool
        elif aggregation == "mean":
            self.pooling = global_mean_pool
        elif aggregation == "max":
            self.pooling = global_max_pool

        self.regression_head = MLP(
            num_layers=num_layers, in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim,
            dropout=dropout, activate_func=activate_func
        )

    def forward(self, x, batch):
        if self.stop_grad:
            x = x.detach()

        x = self.pooling(x, batch)
        x = self.regression_head(x)

        return x


# -------- diffusion related layers --------
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


# -------- model layers --------
class MiniAttentionLayer(torch.nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, d_model, num_heads=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model*num_heads, num_heads, batch_first=True)
        self.qkv_node = torch.nn.Linear(node_dim, d_model * 3 * num_heads)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, d_model * 3 * num_heads)
        self.edge_linear = torch.nn.Sequential(torch.nn.Linear(d_model * num_heads, d_model), 
                                                torch.nn.SiLU(), 
                                                torch.nn.Linear(d_model, out_edge_dim))
    def forward(self, node_us, node_vs, edges):

        # node_us/vs: (B, D)
        q_node_us, k_node_us, v_node_us = self.qkv_node(node_us).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_node_vs, k_node_vs, v_node_vs = self.qkv_node(node_vs).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_edges, k_edges, v_edges = self.qkv_edge(edges).chunk(3, -1) # (B, D*num_heads) for q/k/v

        q = torch.stack([q_node_us, q_node_vs, q_edges], 1) # (B, 3, D*num_heads)
        k = torch.stack([k_node_us, k_node_vs, k_edges], 1) # (B, 3, D*num_heads)
        v = torch.stack([v_node_us, v_node_vs, v_edges], 1) # (B, 3, D*num_heads)

        h, _ = self.multihead_attn(q, k, v)
        h_edge = h[:, -1, :]
        h_edge = self.edge_linear(h_edge)

        return h_edge


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0., activate_func=F.relu):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        if num_layers == 1:
            self.linears.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.linears.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.linears.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.num_layers = num_layers
        self.activate_func = activate_func

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.linears[:-1]:
            x = lin(x)
            x = self.activate_func(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linears[-1](x)

        return x

# class MLP(torch.nn.Module):
#     def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu):
#         super(MLP, self).__init__()

#         self.linear_or_not = True  # default is linear model
#         self.num_layers = num_layers
#         self.use_bn = use_bn
#         self.activate_func = activate_func

#         if num_layers < 1:
#             raise ValueError("number of layers should be positive!")
#         elif num_layers == 1:
#             # Linear model
#             self.linear = torch.nn.Linear(input_dim, output_dim)
#         else:
#             # Multi-layer model
#             self.linear_or_not = False
#             self.linears = torch.nn.ModuleList()

#             self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
#             for layer in range(num_layers - 2):
#                 self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
#             self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

#             if self.use_bn:
#                 self.batch_norms = torch.nn.ModuleList()
#                 for layer in range(num_layers - 1):
#                     self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


#     def forward(self, x):
#         if self.linear_or_not:
#             # If linear model
#             return self.linear(x)
#         else:
#             # If MLP
#             h = x
#             for layer in range(self.num_layers - 1):
#                 h = self.linears[layer](h)
#                 if self.use_bn:
#                     h = self.batch_norms[layer](h)
#                 h = self.activate_func(h)
#             return self.linears[self.num_layers - 1](h)