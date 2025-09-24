from typing import Optional, Dict
from collections.abc import Iterable

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_geometric.utils import degree, scatter, to_dense_batch
from torch_geometric.nn.norm import GraphNorm, PairNorm, MessageNorm, DiffGroupNorm, InstanceNorm, LayerNorm, GraphSizeNorm, MessageNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MessagePassing
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from models.utils import create_activation


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()
nn_act = torch.nn.ReLU() #ReLU()
F_act = F.relu

# -------- layers for sub-graph based node classification --------
def global_mul_pool(x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None):
    dim = -1 if isinstance(x, torch.Tensor) and x.dim() == 1 else -2
    return scatter(x, batch, dim=dim, dim_size=size, reduce='mul')

def creat_pooling_layers(config: Dict):
    ''' Create pooling layers from config. If lengths of arguments differ, extend to same length. '''
    iterable_keys = [k for k in config.keys() if isinstance(config[k], Iterable) and not isinstance(config[k], str)]
    if len(iterable_keys) > 0:
        other_keys = list(set(config.keys()) - set(iterable_keys))
        num_layerss = max([len(config[k]) for k in iterable_keys])
        for k in other_keys:
            # repeat values in other_keys
            config[k] = [config[k]] * num_layerss
        for k in iterable_keys:
            # If lengths of arguments differ, extend to same length.
            if len(config[k]) < num_layerss:
                config[k] = list(config[k]) + [config[k][-1]] * (num_layerss - len(config[k]))

        layers = []
        for i in range(num_layerss):
            layer_config = {k: v[i] for k,v in config.items()}
            layers.append(PoolingLayer(**layer_config))
        return layers
    else:
        return [PoolingLayer(**config)]

class PoolingLayer:
    def __init__(self, pool='sum', subset=False, k=1, subset_type='top', **kwargs):
        """
        Pooling layer supports subseting

        Args:
            subset (bool): flag for subseting
            k (int): number of samples per batch after subseting
            subset_type (str): principle of subseting. 
                'top': top k samples following in-batch order
                'bottom': bottom k samples following in-batch order
                'random': random k samples
                'idx': pre-defined in-batch index (current not implemented)
        """
        self.subset = subset
        self.subset_type = subset_type
        self.k = k
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == 'mul':
            self.pool = global_mul_pool
        else:
            raise NotImplementedError("Invalid graph pooling type.")
        
    def get_idx(self, k, n_sample, idx=None, principle='top', device='cuda:0'):
        assert k <= n_sample
        if principle == 'top':
            return torch.arange(k, device=device)
        elif principle == 'bottom':
            return torch.arange(n_sample - 1, n_sample - k - 1, step=-1, device=device)
        elif principle == 'random':
            return torch.randint(0, n_sample, (k,), dtype=torch.long, device=device)
        elif principle == 'idx':
            assert idx is not None
            return torch.LongTensor(idx, device=device)
        else:
            raise NotImplementedError(f'Unsupported principle {principle}')

    def __call__(self, x, batch, size=None, subset=None, k=None, subset_type=None):
        # overwrite the subseting settings if not None
        subset = subset if subset is not None else self.subset
        if subset:
            k = k if k is not None else self.k
            subset_type = subset_type if subset_type is not None else self.subset_type

            assert x.dim() > 1
            dim = -2
            x, x_mask = to_dense_batch(x, batch)
            n_sample = x.shape[dim]
            subset_idx = self.get_idx(k, n_sample, principle=subset_type, device=x.device)

            x = torch.index_select(x, dim, subset_idx)
            x_mask = torch.index_select(x_mask, dim + 1, subset_idx)
            x = x[x_mask]

            batch_dim = -1
            batch, batch_mask = to_dense_batch(batch, batch)
            batch = torch.index_select(batch, batch_dim, subset_idx)
            batch_mask = torch.index_select(batch_mask, batch_dim, subset_idx)
            batch = batch[batch_mask]
        return self.pool(x, batch, size)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layerss, dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()

        if num_layerss == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layerss - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.num_layerss = num_layerss

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):     
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x


class GINConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        # TODO: check typical GIN setting
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2*in_dim), torch.nn.BatchNorm1d(2*in_dim), nn_act,
            torch.nn.Linear(2*in_dim, out_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            if len(edge_attr.shape) == 1:
                return F_act(x_j * edge_attr.view(-1, 1))
            elif len(edge_attr.shape) == len(x_j.shape) and edge_attr.shape[-1] == x_j.shape[-1]:
                return F_act(x_j + edge_attr)
            else:
                warnings.warn(
                    "Please check the dimension of edge_attr and implement weighted version",
                    stacklevel=2,
                )
                return F_act(x_j)
        else:
            return F_act(x_j)
        

    def update(self, aggr_out):
        return aggr_out

# ### GCN convolution along the graph structure
# class GCNConv(MessagePassing):
#     def __init__(self, in_dim, out_dim):
#         super(GCNConv, self).__init__(aggr='add')

#         self.linear = torch.nn.Linear(in_dim, out_dim)
#         self.root_emb = torch.nn.Embedding(1, out_dim)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.linear(x)

#         row, col = edge_index
#         if len(row) > 0:
#             deg = degree(row, x.size(0), dtype = x.dtype) + 1
#             deg_inv_sqrt = deg.pow(-0.5)
#             deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#             norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#         else:
#             norm = 1

#         return self.propagate(edge_index, x=x, edge_attr = edge_attr, norm=norm) + \
#             F_act(x + self.root_emb.weight) * 1./deg.view(-1,1)

#     def message(self, x_j, edge_attr, norm):
#         if edge_attr is not None:
#             if len(edge_attr.shape) == 1:
#                 return norm.view(-1, 1) * F_act(x_j * edge_attr.view(-1, 1))
#             elif len(edge_attr.shape) == len(x_j.shape) and edge_attr.shape[-1] == x_j.shape[-1]:
#                 return norm.view(-1, 1) * F_act(x_j + edge_attr)
#             else:
#                 warnings.warn(
#                     "Please check the dimension of edge_attr and implement weighted version",
#                     stacklevel=2,
#                 )
#                 return norm.view(-1, 1) * F_act(x_j)
#         else:
#             return norm.view(-1, 1) * F_act(x_j)

#     def update(self, aggr_out):
#         return aggr_out


# a vanilla message passing layer 
class PureConv(nn.Module):
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x


### GNN to generate node embedding
class GNNNode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, input_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False,
                 gnn_name='gin', norm_layer='batch_norm', output_dim=None, act='relu', 
                 act_last=False, encode_raw=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers

        '''

        super(GNNNode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer
        self.act_last = act_last
        self.F_act = create_activation(act)
        self.disable_res_flag = True if output_dim is not None and output_dim != emb_dim else False

        JK = 'last' if self.disable_res_flag == True else JK
        self.JK = JK
        if self.JK == 'stack':
            n_stacked = num_layers + 1 if encode_raw else num_layers
            self.register_parameter("jkparams", torch.nn.Parameter(torch.randn((n_stacked,))))
        # if JK == 'sum' and self.num_layers < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        # self.atom_encoder = AtomEncoder(emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim)
        self.encode_raw = encode_raw
        if self.encode_raw:
            self.proj = MLP(input_dim, emb_dim, emb_dim, 1, drop_ratio)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == num_layers - 1 and output_dim is not None:
                in_dim = input_dim if layer == 0 and not encode_raw else emb_dim
                out_dim = output_dim
            elif layer == 0 and not encode_raw:
                in_dim = input_dim
                out_dim = emb_dim
            else:
                in_dim = out_dim = emb_dim
            if gnn_name == 'gin':
                self.convs.append(GINConv(in_dim, out_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_name == 'sage':
                self.convs.append(SAGEConv(in_dim, out_dim))
            elif gnn_name == 'gat':
                self.convs.append(GATConv(in_dim, out_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_name}')

            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'none':
                    self.batch_norms.append(torch.nn.Identity())
                elif norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(
                        torch.nn.BatchNorm1d(out_dim, track_running_stats=False, affine=False)
                    )
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(out_dim))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(out_dim))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(out_dim))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(out_dim))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(out_dim, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if len(norm_layer.split('_')) > 1 and norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        if self.encode_raw:
            # h_list = [self.atom_encoder(x)]
            # edge_attr = self.bond_encoder(edge_attr)
            h_list = [self.proj(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            h = F.dropout(h_list[layer], self.drop_ratio, training = self.training)
            h = self.convs[layer](h, edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if not (layer == self.num_layers - 1 and not self.act_last):
                h = self.F_act(h)
            if self.residual:
                lower = 0 if self.encode_raw else 1
                if layer >= lower and not (layer == self.num_layers - 1 and self.disable_res_flag):
                    h = h + h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        lower = 0 if self.encode_raw else 1
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(lower, self.num_layers + 1):
                node_representation += h_list[layer]
        elif self.JK == 'stack':
            jkx = torch.stack(h_list[lower:], dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            node_representation = torch.sum(jkx*sftmax, dim=0)
        else:
            raise NotImplementedError(f'Unsupported JK type {self.JK}')

        return node_representation, h_list


### Virtual GNN to generate node embedding
class GNNNodeVirtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, input_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False,
                 gnn_name='gin', norm_layer='batch_norm', output_dim=None, act='relu', 
                 act_last=False, encode_raw=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNNNodeVirtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer
        self.act_last = act_last
        self.F_act = create_activation(act)
        self.disable_res_flag = True if output_dim is not None and output_dim != emb_dim else False

        JK = 'last' if self.disable_res_flag == True else JK
        self.JK = JK
        if self.JK == 'stack':
            n_stacked = num_layers + 1 if self.encode_raw else num_layers
            self.register_parameter("jkparams", torch.nn.Parameter(torch.randn((n_stacked,))))

        # self.atom_encoder = AtomEncoder(emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim)
        self.encode_raw = encode_raw
        if self.encode_raw:
            self.proj = MLP(input_dim, emb_dim, emb_dim, 1, drop_ratio)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == num_layers - 1 and output_dim is not None:
                in_dim = emb_dim
                out_dim = output_dim
            elif layer == 0 and not encode_raw:
                in_dim = input_dim
                out_dim = emb_dim
            else:
                in_dim = out_dim = emb_dim
            if gnn_name == 'gin':
                self.convs.append(GINConv(in_dim, out_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_name == 'sage':
                self.convs.append(SAGEConv(in_dim, out_dim))
            elif gnn_name == 'gat':
                self.convs.append(GATConv(in_dim, out_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_name}')

            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(
                        torch.nn.BatchNorm1d(out_dim, track_running_stats=False, affine=False)
                    )
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(out_dim))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(out_dim))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(out_dim))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(out_dim))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(out_dim, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()
        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act,
                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), nn_act
                )
            )


    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        if self.encode_raw:
            # h_list = [self.atom_encoder(x)]
            # edge_attr = self.bond_encoder(edge_attr)
            h_list = [self.proj(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if layer == self.num_layers - 1 and not self.act_last:
                #remove activation for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = self.F_act(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)

            if self.residual:
                lower = 0 if self.encode_raw else 1
                if layer >= lower and not (layer == self.num_layers - 1 and self.disable_res_flag):
                    h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), 
                                                      self.drop_ratio, training = self.training)


        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            lower = 0 if self.encode_raw else 1
            for layer in range(lower, self.num_layers + 1):
                node_representation += h_list[layer]
        elif self.JK == 'stack':
            jkx = torch.stack(h_list[lower:], dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        else:
            raise NotImplementedError(f'Unsupported JK type {self.JK}')

        return node_representation, h_list



class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

            
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding

# -------- works for graph property prediction / sub-graph based node & link --------
class GNNWithPooling(torch.nn.Module):
    def __init__(self, num_tasks, num_layers=5, input_dim=100, emb_dim=300, gnn_type='gin', 
                 drop_ratio=0.5, norm_layer='batch_norm', residual=True, JK='last', graph_pooling="max", 
                 k=1, subset=False, subset_type='top', predictor_type=None, pred_layers=2, pred_dp=0., 
                 pred_edp=0., pred_ln=None, cndeg=-1, use_xlin=False, tailact=False, twolayerlin=False,
                 beta=1.0, alpha=1.0, scale=5, offset=3, trainresdeg=8, testresdeg=128, pt=0.5,
                 learnablept=False, depth=1, splitsize=-1, encode_raw=False, **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        # JK = 'last'
        # if JK == 'sum' and self.num_layers < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        output_dim = num_tasks if predictor_type is None else None
        act_last = True if predictor_type is not None else False
        gnn_name = gnn_type.split('-')[0]
        if 'virtual' in gnn_type:
            self.graph_encoder = GNNNodeVirtualnode(num_layers, input_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                                    residual=residual, gnn_name=gnn_name, norm_layer=norm_layer,
                                                    output_dim=output_dim, act_last=act_last, encode_raw=encode_raw)
        else:
            self.graph_encoder = GNNNode(num_layers, input_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                         residual=residual, gnn_name=gnn_name, norm_layer=norm_layer, act_last=act_last,
                                         output_dim=output_dim, encode_raw=encode_raw)

        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling is not None:
            pooling_config = {'pool': graph_pooling, 'subset': subset, 'k': k, 'subset_type': subset_type}
            self.pooling_layers = creat_pooling_layers(pooling_config)  # list of pooling layers
        else:
            self.pooling_layers = [lambda x, y: x]

        ### Predictor layers
        self.predictor_type = predictor_type
        if predictor_type == 'ffn':
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2*emb_dim),
                torch.nn.BatchNorm1d(2*emb_dim), 
                torch.nn.ReLU(), 
                torch.nn.Dropout(drop_ratio), 
                torch.nn.Linear(2*emb_dim, self.num_tasks)
            )
        elif predictor_type == 'ncn':
            self.predictor = CNLinkPredictor(emb_dim, emb_dim, 1, pred_layers, pred_dp, pred_edp, pred_ln,
                                             cndeg=cndeg, use_xlin=use_xlin, tailact=tailact, beta=beta,
                                             twolayerlin=twolayerlin)
        elif predictor_type == 'ncnc':
            self.predictor = IncompleteCN1Predictor(emb_dim, emb_dim, 1, pred_layers, pred_dp, pred_edp, pred_ln,
                                                    cndeg=cndeg, use_xlin=use_xlin, tailact=tailact, beta=beta,
                                                    twolayerlin=twolayerlin, alpha=alpha, scale=scale, offset=offset,
                                                    trainresdeg=trainresdeg, testresdeg=testresdeg, pt=pt,
                                                    learnablept=learnablept, depth=depth, splitsize=splitsize)
        else:
            self.predictor = torch.nn.Identity()


    def forward(self, batched_data, first_pooling_only=True):
        h_node, _ = self.graph_encoder(batched_data)
        if first_pooling_only:
            h_graph = self.pooling_layers[0](h_node, batched_data.batch)
        else:
            h_graph = []
            for pool in self.pooling_layers:
                h_graph.append(pool(h_node, batched_data.batch).unsqueeze(1))
            h_graph = torch.cat(h_graph, dim=1).view(-1, h_node.shape[1])
        pred_logits = self.predictor(h_graph)
        return pred_logits, h_graph


# -------- archived --------
class DGCNN(torch.nn.Module):
    def __init__(self, input_dim=100, emb_dim=300, num_layers=5, max_z=1000, k=30, gnn_type='gcn', 
                 drop_ratio=0.5, norm_layer='batch_norm', JK='last', act='tanh', encode_raw=False, 
                 use_feature=False, node_embedding=None, predictor_type='cnn', pred_layers=2, pred_dp=0., 
                 pred_edp=0., pred_ln=None, cndeg=-1, use_xlin=False, tailact=False, twolayerlin=False,
                 beta=1.0, alpha=1.0, scale=5, offset=3, trainresdeg=8, testresdeg=128, pt=0.5,
                 learnablept=False, depth=1, splitsize=-1, **kwargs):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.k = int(k)
        self.max_z = max_z
        self.z_embedding = torch.nn.Embedding(self.max_z, emb_dim)

        initial_channels = emb_dim
        if self.use_feature:
            initial_channels += input_dim
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        gnn_name = gnn_type.split('-')[0]
        output_dim = 1 if predictor_type == 'cnn' else emb_dim
        if 'virtual' in gnn_type:
            self.graph_encoder = GNNNodeVirtualnode(num_layers, initial_channels, emb_dim, JK=JK, 
                                                    drop_ratio=drop_ratio, residual=True, act=act,
                                                    gnn_name=gnn_name, norm_layer=norm_layer, act_last=True,
                                                    output_dim=output_dim, encode_raw=encode_raw)
        else:
            self.graph_encoder = GNNNode(num_layers, initial_channels, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                         residual=True, gnn_name=gnn_name, norm_layer=norm_layer, act=act,
                                         output_dim=output_dim, act_last=True, encode_raw=encode_raw)

        self.predictor_type = predictor_type
        if predictor_type == 'cnn':
            conv1d_channels = [16, 32]
            total_latent_dim = emb_dim * (num_layers - 1) + 1
            conv1d_kws = [total_latent_dim, 5]
            self.predictor = CNNLinkPredictor(self.k, conv1d_channels, conv1d_kws)
        elif predictor_type == 'ncn':
            self.predictor = CNLinkPredictor(emb_dim, emb_dim, 1, pred_layers, pred_dp, pred_edp, pred_ln,
                                             cndeg=cndeg, use_xlin=use_xlin, tailact=tailact, beta=beta,
                                             twolayerlin=twolayerlin)
        elif predictor_type == 'ncnc':
            self.predictor = IncompleteCN1Predictor(emb_dim, emb_dim, 1, pred_layers, pred_dp, pred_edp, pred_ln,
                                                    cndeg=cndeg, use_xlin=use_xlin, tailact=tailact, beta=beta,
                                                    twolayerlin=twolayerlin, alpha=alpha, scale=scale, offset=offset,
                                                    trainresdeg=trainresdeg, testresdeg=testresdeg, pt=pt,
                                                    learnablept=learnablept, depth=depth, splitsize=splitsize)
        else:
            raise NotImplementedError(f'Unsupported predictor type {predictor_type}')

    def forward(self, batched_data, node_id=None):
        z, x, batch = batched_data.z, batched_data.x, batched_data.batch

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        batched_data.x = x
        h, hs = self.graph_encoder(batched_data)
        if self.predictor_type == 'cnn':
            pred = self.predictor(torch.cat(hs[1:], dim=-1), batch)
        else:
            adj = SparseTensor.from_edge_index(
                batched_data.edge_index, sparse_sizes=(batched_data.num_nodes, batched_data.num_nodes)
            ).to_symmetric().coalesce()
            center_idx = torch.arange(
                batched_data.num_nodes, device=h.device
            )[batched_data.center_flag].view(-1,2).T
            pred = self.predictor(h, adj, center_idx).view(-1)

        return pred, hs


# Vanilla MPNN composed of several layers.
class SparseGNN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()
        
        self.adjdrop = DropAdj(edrop)
        
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin or num_layers==0:
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        
        if conv_fn == 'gcn':
            convfn = GCNConv
        elif conv_fn == 'sage':
            convfn = SAGEConv
        elif conv_fn == 'gat':
            convfn = GATConv
        else:
            raise NotImplementedError(f'Unsupported conv_fn {conv_fn}')
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if not noinputlin:
            self.convs.append(PureConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))

        self.lins.append(
            nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                        nn.ReLU(True)))
        for i in range(num_layers - 1):
            self.convs.append(
                convfn(
                    hidden_channels,
                    hidden_channels if i == num_layers - 2 else out_channels))
            if i < num_layers - 2:
                self.lins.append(
                    nn.Sequential(
                        lnfn(
                            hidden_channels if i == num_layers -
                            2 else out_channels, ln),
                        nn.Dropout(dropout, True), nn.ReLU(True)))
            else:
                self.lins.append(nn.Identity())

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


# modified from https://github.com/sangyx/gtrick/blob/main/benchmark/pyg
class EGINConv(MessagePassing):
    def __init__(self, emb_dim, edge_in_channels=None, mol=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr="add")

        self.mol = mol

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(
            2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            assert edge_in_channels is not None
            self.edge_encoder = nn.Linear(edge_in_channels, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_in_channels=None, mol=False):
        super(EGCNConv, self).__init__(aggr='add')

        self.mol = mol

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            assert edge_in_channels is not None
            self.edge_encoder = nn.Linear(edge_in_channels, emb_dim)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()

        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# modified from https://github.com/sangyx/gtrick/blob/main/benchmark/pyg
class EGNN(nn.Module):

    def __init__(self, hidden_channels, out_channels, node_in_channels=None, edge_in_channels=None, num_layers=5,
                 dropout=0.5, conv_type='gin', mol=False):

        super(EGNN, self).__init__()

        self.mol = mol

        if mol:
            self.node_encoder = AtomEncoder(hidden_channels)
        else:
            if node_in_channels is None:
                self.node_encoder = nn.Embedding(1, hidden_channels)
            else:
                self.node_encoder = nn.Embedding(node_in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.num_layers = num_layers

        for i in range(self.num_layers):
            if conv_type == 'gin':
                self.convs.append(
                    EGINConv(hidden_channels, edge_in_channels, mol))
            elif conv_type == 'gcn':
                self.convs.append(
                    EGCNConv(hidden_channels, edge_in_channels, mol))

            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)

        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

        h = self.node_encoder(x)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_attr)

        if not self.mol:
            h = self.bns[-1](h)

        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)

        h = self.out(h)

        return h


# -------- utils for NCN & NCNC --------
class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret


def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    idx = torch.searchsorted(element1[:-1], element2)
    matchedmask = (element1[idx] == element2)

    maskelem1 = torch.ones_like(element1, dtype=torch.bool)
    maskelem1[idx[matchedmask]] = 0
    retelem1 = element1[maskelem1]

    retelem2 = element2[torch.logical_not(matchedmask)]
    return elem2spm(retelem1, adj1.sizes()), elem2spm(retelem2, adj2.sizes())


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               tarei: Tensor,
               filled1: bool = False,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    # a wrapper for functions above.
    adj1 = adj1[tarei[0]]
    adj2 = adj2[tarei[1]]
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap