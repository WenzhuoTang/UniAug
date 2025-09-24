import scipy.sparse as ssp

import torch
from torch_geometric.nn import MessagePassing, Node2Vec
from torch_geometric.utils import degree

from dataset.heuristics import CN, AA, RA, PPR, katz_apro, katz_close


FEATURE_CHOICE = ['node2vec', 'cn', 'aa', 'ra', 'ppr', 'katz', 'degree']

# -------- generate structure feature --------
def get_csr_adj(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    edge_weight = torch.ones(edge_index.shape[1], dtype=float) if edge_weight is None else edge_weight
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 
    return A

def get_features(edge_index, edge_weight=None, num_nodes=None, feature_types=['all'], 
                 embedding_dim=128, walk_length=20, context_size=20, walks_per_node=1, 
                 p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, 
                 beta_katz=0.005, path_len=3, remove=False, fill_zeros=False, 
                 batch_size=256, lr=1e-2, epochs=100, device='cpu', print_loss=False, 
                 return_dict=False, **kwargs):
    if 'all' in feature_types:
        feature_types = FEATURE_CHOICE

    assert all([x in FEATURE_CHOICE for x in feature_types])
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    A = get_csr_adj(edge_index, edge_weight, num_nodes)
    node_feat_dict = {}
    edge_attr_dict = {}
    if 'node2vec' in feature_types:
        model = Node2Vec(
            edge_index, embedding_dim=embedding_dim, walk_length=walk_length, 
            context_size=context_size, walks_per_node=walks_per_node, p=p_node2vec, 
            q=q_node2vec, num_negative_samples=num_negative_samples, sparse=True
        ).to(device)
        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
        model.train()
        for epoch in range(1, epochs + 1):
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
            if print_loss and epoch % 20 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        feat = model().detach().cpu()
        if len(feat) < num_nodes and fill_zeros:
            zeros = torch.zeros(num_nodes - edge_index.max() - 1, feat.shape[1], dtype=feat.dtype)
            feat = torch.cat([feat, zeros])
        node_feat_dict['node2vec'] = feat

    if 'degree' in feature_types:
        node_feat_dict['degree'] = degree(edge_index[0], num_nodes=num_nodes)

    if 'cn' in feature_types:
        edge_attr_dict['cn'] = CN(A, edge_index)
    if 'aa' in feature_types:
        edge_attr_dict['aa'] = AA(A, edge_index)
    if 'ra' in feature_types:
        edge_attr_dict['ra'] = RA(A, edge_index)
    if 'ppr' in feature_types:
        edge_attr_dict['ppr'] = PPR(A, edge_index, p_ppr)
    if 'katz' in feature_types:
        if num_nodes <= 1e6:
            edge_attr_dict['katz'] = katz_close(A, edge_index, beta_katz)
        else:
            edge_attr_dict['katz'] = katz_apro(A, edge_index, beta_katz, path_len, remove)

    if return_dict:
        node_feat_dict.update(edge_attr_dict)
        return node_feat_dict
    else:
        node_feat = torch.cat(node_feat_dict.values(), dim=1) if len(node_feat_dict) > 0 else None
        edge_attr = torch.cat(edge_attr_dict.values(), dim=1) if len(edge_attr_dict) > 0 else None
        return node_feat, edge_attr

def iteratively_get_features(edge_index, edge_weight=None, num_nodes=None, list_of_kwargs=[]):
    node_feat_list = []
    edge_attr_list = []
    for kwargs in list_of_kwargs:
        node_feat, edge_attr = get_features(edge_index, edge_weight, num_nodes, return_dict=False, **kwargs)
        if node_feat is not None:
            node_feat_list.append(node_feat)
        if edge_attr is not None:
            edge_attr_list.append(edge_attr)

    node_feat = torch.cat(node_feat_list, dim=1) if len(node_feat_list) > 0 else None
    edge_attr = torch.cat(edge_attr_list, dim=1) if len(edge_attr_list) > 0 else None
    return node_feat, edge_attr


# -------- utils for aggregation --------
class AggregateEdgeAttr(MessagePassing):
    def __init__(self, aggr='add', repeat=1):
        super().__init__(aggr=aggr)
        self.repeat = repeat

    def forward(self, x, edge_index, edge_attr):
        agged_edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return torch.cat([x] + [agged_edge_attr] * self.repeat, dim=-1)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr
    
def aggregate_edge_attr(x, edge_index, edge_attr, repeat=1, aggr='add'):
    agg_func = AggregateEdgeAttr(aggr, repeat)
    return agg_func(x, edge_index, edge_attr)