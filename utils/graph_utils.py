import numpy as np
import networkx as nx

from multiprocessing import Pool

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max, scatter_std


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


# -------- Create initial node features --------
def init_features(init, adjs=None, nfeat=10):

    if init=='zeros':
        feature = torch.zeros((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='ones':
        feature = torch.ones((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='deg':
        feature = adjs.sum(dim=-1).to(torch.long)
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            print(feature.max().item())
            raise NotImplementedError(f'max_feat_num mismatch')
    else:
        raise NotImplementedError(f'{init} not implemented')

    flags = node_flags(adjs)

    return mask_x(feature, flags)


# -------- Sample initial flags tensor from the training graph set --------
def init_flags(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])

    return flags


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


# -------- Quantize generated graphs --------
def quantize(adjs, thr=0.5):
    adjs_ = torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))
    return adjs_


# -------- Quantize generated molecules --------
# adjs: 32 x 9 x 9
def quantize_mol(adjs):                         
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def adjs_to_graphs(adjs, is_cuda=False):
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


# -------- Check if the adjacency matrices are symmetric --------
def check_sym(adjs, print_val=False):
    sym_error = (adjs-adjs.transpose(-1,-2)).abs().sum([0,1,2])
    if not sym_error < 1e-2:
        raise ValueError(f'Not symmetric: {sym_error:.4e}')
    if print_val:
        print(f'{sym_error:.4e}')


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


# -------- Create padded adjacency matrices --------
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def graphs_to_tensor(graph_list, max_node_num):
    adjs_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        adj = nx.to_numpy_matrix(g, nodelist=node_list)
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        adjs_list.append(padded_adj)

    del graph_list

    adjs_np = np.asarray(adjs_list)
    del adjs_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    del adjs_np

    return adjs_tensor 


def graphs_to_adj(graph, max_node_num):
    max_node_num = max_node_num

    assert isinstance(graph, nx.Graph)
    node_list = []
    for v, feature in graph.nodes.data('feature'):
        node_list.append(v)

    adj = nx.to_numpy_matrix(graph, nodelist=node_list)
    padded_adj = pad_adjs(adj, node_number=max_node_num)

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    del padded_adj

    return adj


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair

# ------------------------------------------------------------------------------------------
# ------------------------------ utils for graph augmentation ------------------------------
# ------------------------------------------------------------------------------------------

# -------- utils to convert sparse input to dense input --------
def convert_sparse_to_dense(batch_index, node_feature, edge_index, edge_attr=None,
                            augment_mask=None, return_node_mask=True):
    max_count = torch.unique(batch_index, return_counts=True)[1].max()
    dense_adj = to_dense_adj(edge_index, batch=batch_index, edge_attr=edge_attr, max_num_nodes=max_count).to(torch.float32)
    if augment_mask is not None:
        dense_adj = dense_adj[augment_mask]
    # process node feature B, N, F
    dense_x, node_mask = to_dense_batch(node_feature, batch=batch_index, max_num_nodes=max_count)
    if augment_mask is not None:
        dense_x = dense_x[augment_mask]
    if return_node_mask:
        return dense_x, dense_adj, node_mask
    else:
        return dense_x, dense_adj

# -------- get raw graph feature from graph node features and structures --------
def extract_graph_feature(x, adj, node_mask=None, pooling=None):
    if node_mask is None:
        node_mask = (adj.sum(dim=-1)>0).bool()
    valid_num = node_mask.sum(dim=-1, keepdim=True)

    feat_rep = extract_node_feature(x, adj, node_mask, pooling)
    graph_rep = extract_structure_feature(x, adj, node_mask)
    if len(valid_num.size()) != len(feat_rep.size()):
        final_rep = torch.concat([feat_rep, graph_rep], dim=-1)
    else:
        final_rep = torch.concat([valid_num, feat_rep, graph_rep], dim=-1)
    return final_rep

def extract_node_feature(x, adj, node_mask, pooling=None):
    # x: (bs, n, F)
    # adj: (bs, n, n)
    def get_stat(x, batch_index, pooling=None):
        size = batch_index[-1].item() + 1
        results_list = [scatter_mean(x, batch_index, dim=0, dim_size=size), 
                        scatter_min(x, batch_index, dim=0, dim_size=size)[0], 
                        scatter_max(x, batch_index, dim=0, dim_size=size)[0]]
        if results_list[0].shape[1] > 1:
            if pooling == 'mean':
                results_list = [x.mean(-1, keepdim=True) for x in results_list]
            elif pooling == 'sum':
                results_list = [x.sum(-1, keepdim=True) for x in results_list]
            elif pooling == 'max':
                results_list = [x.max(-1, keepdim=True)[0] for x in results_list]

        return torch.cat(results_list, dim=-1)

    batch_index = node_mask.nonzero()[:,0]
    node_attr = x[node_mask]
    node_deg = adj.sum(dim=-1)[node_mask]
    # attr_feat = get_stat(node_attr.view(-1,1), batch_index)
    # deg_feat = get_stat(node_deg.view(-1,1), batch_index)
    attr_feat = get_stat(node_attr.view(len(batch_index),-1), batch_index, pooling)
    deg_feat = get_stat(node_deg.view(len(batch_index),-1), batch_index, pooling)
    attr_feat = attr_feat / (attr_feat.norm(dim=0, keepdim=True) + 1e-18)
    deg_feat = deg_feat / (deg_feat.norm(dim=0, keepdim=True) + 1e-18)
    return torch.cat([attr_feat, deg_feat], dim=-1)

def extract_structure_feature(x, adj, node_mask):
    batch_index = node_mask.nonzero()[:,0]
    ret_feat = scatter_add(adj.sum(dim=-1,keepdim=True)[node_mask], batch_index, dim=0, dim_size=x.size(0)) # degree
    ret_feat = torch.cat([ret_feat, scatter_add(x[node_mask], batch_index, dim=0, dim_size=x.size(0))], dim=-1)  # feat distribution
    return ret_feat

# -------- dense input -> sparse input --------
def convert_dense_adj_to_sparse_with_attr(adj, node_mask=None): # B x N x N
    if node_mask is None:
        node_mask = torch.ones(len(adj), dtype=bool, device=adj.device)
    adj = adj[node_mask]
    edge_index = (adj > 0.5).nonzero().t()
    row, col = edge_index[0], edge_index[1]
    edge_attr = adj[row, col]
    return torch.stack([row, col], dim=0), edge_attr

# -------- dense input -> pyg object --------
def convert_dense_to_rawpyg(dense_x, dense_adj, augmented_labels=None, multiprocess=False, 
                            n_jobs=20, **dense_inputs):
    # dense_x: B, N, F; dense_adj: B, N, N; return: B, N, F, adj
    pyg_graph_list = []
    if isinstance(augmented_labels, torch.Tensor):
        augmented_labels = augmented_labels.cpu()
    dense_x = dense_x.cpu()
    dense_adj = dense_adj.cpu()
    input_keys = sorted(dense_inputs.keys())
    
    if multiprocess:
        with Pool(n_jobs) as pool:  # Pool created
            results = pool.map(
                get_pyg_data_from_dense_batch, 
                [
                    (
                        dense_x[i], dense_adj[i], 
                        augmented_labels[i] if augmented_labels is not None else None,
                        {k: dense_inputs[k][i] for k in input_keys} 
                        if len(input_keys) > 0 else {},
                    )
                    for i in range(len(dense_x))
                ]
            )        
        for single_results in results:
            pyg_graph_list.extend(single_results)
    else:
        input_list = [dense_inputs[k] for k in input_keys] if len(input_keys) > 0 else []
        for b_index, inputs in enumerate(zip(dense_x, dense_adj, *input_list)):
            x_single, adj_single = inputs[0], inputs[1]
            edge_index, edge_attr = dense_to_sparse(adj_single)
            temp_dict = dict(zip(input_keys, inputs[2:]))
            temp_dict.update({
                'x': x_single, 'edge_index': edge_index, 'edge_attr': edge_attr
            })
            if 'y' not in temp_dict.keys() and augmented_labels is not None:
                temp_dict['y'] = augmented_labels[b_index].view(1, -1)
            g = Data(**temp_dict)
            pyg_graph_list.append(g)
    return pyg_graph_list

def get_pyg_data_from_dense_batch(params):
    batched_x, batched_adj, augmented_labels, batched_input = params
    input_keys = sorted(batched_input.keys())
    input_list = [batched_input[k] for k in input_keys] if len(input_keys) > 0 else []
    pyg_graph_list = []
    for b_index, inputs in enumerate(zip(batched_x, batched_adj, *input_list)):
        x_single, adj_single = inputs[0], inputs[1]
        edge_index, edge_attr = dense_to_sparse(adj_single)
        temp_dict = dict(zip(input_keys, inputs[2:]))
        temp_dict.update({
            'x': x_single, 'edge_index': edge_index, 'edge_attr': edge_attr
        })
        if 'y' not in temp_dict.keys() and augmented_labels is not None:
            temp_dict['y'] = augmented_labels[b_index].view(1, -1)
        g = Data(**temp_dict)
        pyg_graph_list.append(g)

    return pyg_graph_list