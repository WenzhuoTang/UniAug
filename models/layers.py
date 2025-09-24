from typing import Optional, Iterable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, MessagePassing,
    global_add_pool, global_mean_pool, global_max_pool,
)
from torch_geometric.utils import scatter, to_dense_batch
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


# -------- pooling layer with subset option --------
def global_mul_pool(x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None):
    dim = -1 if isinstance(x, torch.Tensor) and x.dim() == 1 else -2
    return scatter(x, batch, dim=dim, dim_size=size, reduce='mul')


class PoolingLayer:
    def __init__(self, pool='sum', subset=False, subset_type='top', k=1, **kwargs):
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

    def __call__(self, x, batch, size=None):
        # overwrite the subseting settings if not None

        if self.subset:
            k, subset_type = self.k, self.subset_type

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


# -------- GCN layers from NCN --------
class PureConv(nn.Module):
    aggr: Final[str]
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
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}


# -------- GNN with virtual node and without edge encoder --------
class EGINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr=None):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))

        return out

    def message(self, x_j):
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class GINNodeVirtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, edge_feat = False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINNodeVirtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if edge_feat:
                self.convs.append(EGINConv(emb_dim))
            else:
                self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation