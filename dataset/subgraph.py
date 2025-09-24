import copy
import operator
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch_geometric
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.utils import (
    coalesce, subgraph, to_dense_adj,
    to_undirected, add_remaining_self_loops
)
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.transforms import (
    RootedEgoNets,
    RootedRWSubgraph,
)


# -------- extract induced subgraphs --------
def relabel_nodes(edge_index, nodes, num_nodes, center_nodes=None, move_center_nodes=False):
    node_mask = index_to_mask(nodes, size=num_nodes)
    node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                           device=edge_index.device)
    temp = nodes.clone()
    if move_center_nodes:  # move center nodes to the first several nodes
        assert center_nodes is not None
        center_idx = [torch.where(nodes == x)[0].item() for x in center_nodes]
        swap_idx = range(len(center_nodes))
        if not set(center_idx).isdisjoint(swap_idx):
            int_idx = set(center_idx).intersection(swap_idx)
            center_idx = list(set(center_idx) - set(int_idx))
            swap_idx = list(set(swap_idx) - set(int_idx))
        if len(swap_idx) > 0:
            temp[swap_idx], temp[center_idx] = temp[center_idx], temp[swap_idx]
            idx_dict = {'swap': swap_idx, 'center': center_idx}
        else:
            idx_dict = None
    else:
        idx_dict = None
    node_idx[temp] = torch.arange(node_mask.sum().item(), device=edge_index.device)
    return node_idx[edge_index], idx_dict

def relabel_nodes_with_mapping(edge_index, nodes=None, mapping=None):
    if mapping is None:
        assert nodes is not None
        mapping = dict(zip(nodes.cpu().tolist(), range(len(nodes))))
    out_edge_index = edge_index.clone()
    out_edge_index.apply_(lambda x: mapping.__getitem__(x))
    return out_edge_index, mapping

def extract_subgraphs(
        data, y=None, nodes_list=None, subgraph_type='ego', num_hops=2, walk_length=10, 
        repeat=5, max_node_num=100,  # subgraph kwargs
        sampling_mode=None, random_init=False, minimum_redundancy=3, shortest_path_mode_stride=2, 
        random_mode_sampling_rate=0.5,  # subsample kwargs
        fill_zeros=False, add_self_loop=False, move_center_nodes=True, rank_nodes='feat_corr',
        remove_center_edges=False, make_undirected=True, node_feat_dict={}, **kwargs  # other kwargs
    ):
    if subgraph_type == 'ego':
        transform = RootedEgoNets(num_hops=num_hops)
    elif subgraph_type == 'rw':
        transform = RootedRWSubgraph(walk_length=walk_length, repeat=repeat)
    data_transformed = transform(data)
    
    x = data_transformed.x
    num_nodes = data_transformed.num_nodes
    edge_index = data_transformed.edge_index
    subgraphs_nodes = torch.stack([data_transformed.n_sub_batch, data_transformed.n_id])
    subgraphs_edges = torch.stack([data_transformed.e_sub_batch, data_transformed.e_id])

    if nodes_list is None:
        nodes_list = [[x] for x in range(num_nodes)]

    
    data_list = []
    node_coor_list = []
    node_feat_dict.update({'x': x})
    for i in tqdm(range(len(nodes_list))):

        feat_dict = node_feat_dict.copy()
        nodes = nodes_list[i]
        sub_nodes_list = []
        sub_edges_list = []
        for b in nodes:
            sub_nodes_list.append(subgraphs_nodes[:, subgraphs_nodes[0] == b])
            sub_edges_list.append(subgraphs_edges[:, subgraphs_edges[0] == b])

        sub_nodes = remove_duplicates_in_row_k(torch.cat(sub_nodes_list, dim=1), k=1)
        sub_edges = remove_duplicates_in_row_k(torch.cat(sub_edges_list, dim=1), k=1)
        sub_edge_index = data_transformed.edge_index[:, sub_edges[1]]

        # temporary fix: add self-loop when edge_index is empty
        if sub_edge_index.shape[1] == 0:
            sub_edge_index = sub_nodes.to(edge_index.dtype)
        
        if len(nodes) > 1 and remove_center_edges:
            # warning: graph without self-loop may result in empty edge_index
            nodes_comb = list(itertools.combinations(nodes, 2))
            for e in nodes_comb:
                e = torch.tensor(e, dtype=sub_edge_index.dtype).unsqueeze(-1)
                e_flipped = e[list(reversed(range(len(e))))]
                mask = torch.logical_or(sub_edge_index != e, sub_edge_index != e_flipped).all(0)
                sub_edge_index = sub_edge_index[:, mask]

        if max_node_num is None or sub_nodes.shape[1] < max_node_num:
            sub_edge_index, idx_dict = relabel_nodes(sub_edge_index, sub_nodes[1], num_nodes, 
                                                     nodes, move_center_nodes)
            for k in node_feat_dict.keys():
                if node_feat_dict[k] is not None:
                    feat_dict[k] = node_feat_dict[k][sub_nodes[1]].detach().cpu().clone()
                    if max_node_num is not None and fill_zeros:
                        zeros = torch.zeros(max_node_num - sub_nodes.shape[1], 
                                            feat_dict[k].shape[1], 
                                            dtype=feat_dict[k].dtype)
                        feat_dict[k] = torch.cat([feat_dict[k], zeros])

        elif sub_nodes.shape[1] >= max_node_num:
            if rank_nodes == 'random':
                sub_nodes = sub_nodes[:, torch.randperm(sub_nodes.shape[1])[:max_node_num]]
                nodes_tensor = torch.tensor(nodes).repeat(2, 1)
                if not all([
                    (nodes_tensor[:,[i]] == sub_nodes).any(1).all() for i in range(nodes_tensor.shape[1])
                ]):
                    # make sure center nodes are selected
                    sub_nodes = torch.cat([torch.tensor(nodes).repeat(2, 1), sub_nodes], dim=1)
                    sub_nodes = remove_duplicates_in_row_k(sub_nodes, k=1)

                if sub_nodes.shape[1] > max_node_num:
                    sub_nodes = sub_nodes[:, :max_node_num]
            elif rank_nodes == 'feat_corr':
                assert x is not None
                center_idx = [torch.where(sub_nodes[1] == x)[0].item() for x in nodes]
                other_idx = np.array(list(set(range(len(sub_nodes[1]))) - set(center_idx)))
                feat_corr = torch.corrcoef(x)[center_idx][:, other_idx]  # corr between other nodes and center nodes
                rank = torch.argsort(feat_corr.sum(0), descending=True).tolist()  # aggregate rank across center nodes with sum
                out_idx = center_idx + list(other_idx[rank][:(max_node_num - len(center_idx))])
                sub_nodes = sub_nodes[:, out_idx]
            else:
                raise NotImplementedError(f'Unsupported nodes ranking principle {rank_nodes}')

            sub_edge_index, _, edge_mask = subgraph(sub_nodes[1], sub_edge_index, num_nodes=num_nodes, 
                                                    relabel_nodes=False, return_edge_mask=True)
            sub_edge_index, idx_dict = relabel_nodes(sub_edge_index, sub_nodes[1], num_nodes, 
                                                        nodes, move_center_nodes)
            sub_edges = sub_edges[:, edge_mask]

            for k in node_feat_dict.keys():
                if node_feat_dict[k] is not None:
                    feat_dict[k] = node_feat_dict[k][sub_nodes[1]].detach().cpu().clone()
        
        if idx_dict is not None:
            swap_idx, center_idx = idx_dict['swap'], idx_dict['center']
            for k in feat_dict.keys():
                if feat_dict[k] is not None:
                    feat_dict[k][swap_idx], feat_dict[k][center_idx] = \
                        feat_dict[k][center_idx], feat_dict[k][swap_idx]

        node_coor_list.append(sub_nodes)
        # edge_coor_list.append(sub_edges)
        data_dict = {
            'edge_index': sub_edge_index,
            **feat_dict
        }
        if y is not None:
            data_dict['y'] = y[i]

        if data_transformed.edge_attr is not None:
            data_dict['edge_attr'] = data_transformed.edge_attr[sub_edges[1]]
        else:
            data_dict['edge_attr'] = None

        if add_self_loop:
            data_dict['edge_index'], data_dict['edge_attr'] = add_remaining_self_loops(
                data_dict['edge_index'], data_dict['edge_attr']
            )

        if make_undirected:
            data_dict['edge_index'], data_dict['edge_attr'] = to_undirected(
                data_dict['edge_index'], data_dict['edge_attr']
            )

        data_list.append(Data(**data_dict))

    if sampling_mode is not None:
        print(f'Subsample subgraphs with mode {sampling_mode}')
        selected_index, node_selected_times = subsampling_subgraphs(
            edge_index, node_coor_list, num_nodes, nodes_list, sampling_mode,
            random_init, minimum_redundancy, shortest_path_mode_stride, 
            random_mode_sampling_rate
        )
        print(f'{len(selected_index)} subgraphs are selected.')
        print(f'node selected times:  {node_selected_times.mean():.4f} +/- {node_selected_times.std():.4f}\
                min = {node_selected_times.min()}, max = {node_selected_times.max()}.')

        import operator
        f = operator.itemgetter(*selected_index)
        data_list = list(f(data_list))

    return data_list

# -------- extract graph segments (METIS) with extension to k-hop neighbors (from MLPMixer) --------
# Currently is built on undirected graphs
def extract_segments(data, thres=200, return_list=True, fill_zeros=False, max_node_num=None, add_node_id=True,
                     add_edge_id=False, num_hops=0):
    N, E = data.num_nodes, data.num_edges
    adj = SparseTensor(
        row=data.edge_index[0], col=data.edge_index[1],
        value= torch.arange(E, device=data.edge_index.device),
        sparse_sizes=(N, N)
    )
    num_partition = N // thres + 1
    N_PART_MAX = int(1e4)

    if num_partition > N_PART_MAX:
        # adj = adj.to_symmetric()  # this may take forever if graph is too large
        _num_partition = num_partition // N_PART_MAX + 1
        _, _partptr, _perm = adj.partition(_num_partition, False)
        _part_list = [torch.arange(_partptr[i], _partptr[i+1]) for i in range(len(_partptr) - 1)]
        _rev_perm = torch.argsort(_perm).tolist()
        _rev_mapping = dict(zip(_rev_perm, range(N)))

        part_list = []
        for p in tqdm(_part_list):
            p.apply_(lambda x: _rev_mapping.__getitem__(x))
            sub_adj = adj[p, p]
            sub_num_partition = len(p) // thres + 1
            _, sub_partptr, sub_perm = sub_adj.partition(sub_num_partition, False)
            sub_part_list = [torch.arange(sub_partptr[i], sub_partptr[i+1]) 
                            for i in range(len(sub_partptr) - 1)]
            sub_rev_perm = torch.argsort(sub_perm).tolist()
            sub_rev_mapping = dict(zip(sub_rev_perm, range(N)))
            for p_sub in sub_part_list:
                p_sub.apply_(lambda x: sub_rev_mapping.__getitem__(x))
            part_list.extend(sub_part_list)

    else:
        adj = adj.to_symmetric()
        adj, partptr, perm = adj.partition(num_partition, False)
        part_list = [torch.arange(partptr[i], partptr[i+1]) for i in range(len(partptr) - 1)]
        rev_perm = torch.argsort(perm).tolist()
        rev_mapping = dict(zip(rev_perm, range(N)))
        for p in part_list:
            p.apply_(lambda x: rev_mapping.__getitem__(x))

    data.part_list = part_list
    # extend to k hop neighbors, may OOM if num_nodes is large
    if num_hops > 0:
        # transform = RootedEgoNets(num_hops=num_hops)
        # data_transformed = transform(data)
        # subgraphs_nodes = torch.stack([data_transformed.n_sub_batch, data_transformed.n_id])

        # extended_part_list = []
        # for nodes in part_list:
        #     sub_nodes_list = []
        #     for node in nodes:
        #         sub_nodes_list.append(subgraphs_nodes[1, subgraphs_nodes[0] == node])
        #     sub_nodes = torch.cat(sub_nodes_list).unique()
        #     extended_part_list.append(sub_nodes)

        k_hop_node_mask = k_hop_subgraph(data.edge_index, data.num_nodes, num_hops)
        extended_part_list = [torch.where(k_hop_node_mask[p].any(0))[0] for p in part_list]
        data.part_list = extended_part_list

    if return_list:
        max_node_num = max_node_num if max_node_num is not None else thres
        return graph_to_segments(data, add_node_id, add_edge_id, fill_zeros, max_node_num, 
                                 return_remaining=False)
    else:
        return data

def graph_to_segments(data, add_node_id=True, add_edge_id=False, fill_zeros=False, max_node_num=None,
                      return_remaining=True, add_diff_attr=False, node_attr_keys=None, edge_attr_keys=None,
                      graph_attr_keys=None):
    # XXX: requires part_list of segments
    N, part_list = data.num_nodes, data.part_list
    data_list = []
    if return_remaining:
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=bool, device=data.edge_index.device)

    for nodes in part_list:
        data_dict = {}
        data_dict['edge_index'], data_dict['edge_attr'], _edge_mask = subgraph(
            nodes, data.edge_index, data.edge_attr, num_nodes=N, relabel_nodes=False, return_edge_mask=True
        )
        data_dict['edge_index'], node_mapping = relabel_nodes_with_mapping(
            data_dict['edge_index'], nodes=nodes
        )
        if return_remaining:
            edge_mask = torch.logical_or(edge_mask, _edge_mask)
        if add_edge_id:
            data_dict['edge_id'] = torch.where(_edge_mask)[0]
        if data.x is not None:
            data_dict['x'] = data.x[nodes]
        if data.y is not None:
            data_dict['y'] = data.y[nodes]
        if add_node_id:
            data_dict['node_id'] = nodes

        if node_attr_keys is not None:
            for k in node_attr_keys:
                data_dict[k] = data[k][nodes]

        if edge_attr_keys is not None:
            for k in edge_attr_keys:
                data_dict[k] = data[k][_edge_mask]
        
        if graph_attr_keys is not None:
            for k in graph_attr_keys:
                data_dict[k] = data[k]

        if fill_zeros:
            x = data_dict['x']
            zeros = torch.zeros(max_node_num - x.shape[0], x.shape[1], 
                                dtype=x.dtype, device=x.device)
            data_dict['x'] = torch.cat([x, zeros])
            if data.y is not None:
                neg_fill = -1 * torch.ones(max_node_num - x.shape[0], *data.y.shape[1:], dtype=data.y.dtype,
                                           device=x.device)
                data_dict['y'] = torch.cat([data_dict['y'], neg_fill])
            if add_node_id:
                neg_fill = -1 * torch.ones(max_node_num - x.shape[0], dtype=nodes.dtype,
                                        device=x.device)
                data_dict['node_id'] = torch.cat([data_dict['node_id'], neg_fill])

        temp_data = Data(**data_dict).coalesce()
        if add_diff_attr:
            adj = to_dense_adj(temp_data.edge_index, max_num_nodes=temp_data.num_nodes)[0].long()
            row, col = torch.triu_indices(temp_data.num_nodes, temp_data.num_nodes, 1)
            temp_data.full_edge_index = torch.stack([row, col])
            temp_data.full_edge_attr = adj[temp_data.full_edge_index[0], temp_data.full_edge_index[1]]
            temp_data.nodes_per_graph = temp_data.num_nodes
            temp_data.edges_per_graph = temp_data.num_nodes * (temp_data.num_nodes - 1) // 2

        data_list.append(temp_data)

    if return_remaining:
        remaining_edge_index = data.edge_index[:, ~edge_mask]
        if data.edge_attr is not None:
            remaining_edge_attr = data.edge_attr[~edge_mask]
        else:
            remaining_edge_attr = None
        return data_list, remaining_edge_index, remaining_edge_attr
    else:
        return data_list

def segments_to_graph(data_list, remaining_edge_index=None, remaining_edge_attr=None, reduce='max'):
    # XXX: currently does not support edge_attr
    # XXX: requires node_id for each segment
    node_ids, edge_index_list, edge_attr_list, x_list, y_list = [], [], [], [], []
    
    x_list = []
    y_list = []
    for data in data_list:
        x, y = data.x, data.y
        node_id = data.node_id
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)

        # map back to node_id
        mask = node_id >= 0
        node_id = node_id[mask]
        N = mask.sum().item()
        mapping = dict(zip(range(N), node_id.cpu().tolist()))

        # remove links of virtual nodes
        edge_mask = (edge_index < N).all(0)
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]

        edge_index.apply_(lambda x: mapping.__getitem__(x))
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        node_ids.append(node_id)
        if x is not None:
            x_list.append(x[mask])
        if y is not None:
            y_list.append(y[mask])

    # rev_perm = torch.argsort(torch.cat(node_ids))
    node_ids = torch.cat(node_ids)

    if remaining_edge_attr is not None and remaining_edge_attr.shape[0] > 0:
        remaining_edge_attr = remaining_edge_attr.to(node_ids.device)
        edge_attr_list.append(remaining_edge_attr)
    if remaining_edge_index is not None and remaining_edge_index.shape[1] > 0:
        remaining_edge_index = remaining_edge_index.to(node_ids.device)
        edge_index_list.append(remaining_edge_index)
        if remaining_edge_attr is None and edge_attr_list[-1] is not None:
            size = list(edge_attr_list[-1].size())
            size[0] = remaining_edge_index.shape[1]
            fill = torch.ones(size, dtype=edge_attr_list[-1].dtype,
                              device=edge_attr_list[-1].device)
            edge_attr_list.append(fill)

    data_dict = {'edge_index': torch.cat(edge_index_list, dim=1)}
    if len(edge_attr_list) > 0:
        data_dict['edge_attr'] = torch.cat(edge_attr_list, dim=-1)
    if len(x_list) > 0:
        data_dict['x'] = scatter(torch.cat(x_list), node_ids, dim=0, reduce=reduce)
    if len(y_list) > 0:
        data_dict['y'] = scatter(torch.cat(y_list), node_ids, dim=0, reduce=reduce)

    data = Data(**data_dict)
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, data.edge_attr, num_nodes=data.num_nodes, reduce=reduce
    )
    return data

def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # each one contains <= i hop masks
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask

# -------- utils for subsampling subgraphs --------
def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse+decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, inverse, counts, index

def remove_duplicates_in_row_k(x, k=0):
    _, _, _, index = unique(x[k])
    return x[:, index]

def subsampling_subgraphs(edge_index, node_coor_list, num_nodes, center_node_list, sampling_mode='shortest_path', 
                          random_init=False, minimum_redundancy=3, shortest_path_mode_stride=2,
                          random_mode_sampling_rate=0.5):

    if sampling_mode == 'random': 
        selected_index, node_selected_times = random_sampling(
            node_coor_list, num_nodes, rate=random_mode_sampling_rate, 
            minimum_redundancy=minimum_redundancy
        )
    elif sampling_mode == 'shortest_path':
        selected_index, node_selected_times = shortest_path_sampling(
            edge_index, node_coor_list, num_nodes, center_node_list, minimum_redundancy=minimum_redundancy,
            stride=max(1, shortest_path_mode_stride), random_init=random_init
        )
    # elif sampling_mode in ['min_set_cover']:
    #     assert subgraphs_nodes.size(0) == num_nodes # make sure this is subgraph_nodes_masks
    #     selected_index, node_selected_times = min_set_cover_sampling(
    #         edge_index, subgraphs_nodes, minimum_redundancy=minimum_redundancy, random_init=random_init
    #     )
    else:
        raise NotImplementedError(f'Unsupported sampling_mode {sampling_mode}')

    return selected_index, node_selected_times

def random_sampling(node_coor_list, num_nodes, rate=0.5, minimum_redundancy=0, increase_rate=False,
                    increase_interval=10):
    n_graphs = len(node_coor_list)
    counter = 0
    while True:
        counter += 1
        selected = np.random.choice(n_graphs, int(n_graphs*rate), replace=False)
        f = operator.itemgetter(*selected)
        selected_node_coor = torch.cat(list(f(node_coor_list)), dim=1)
        node_selected_times = torch.bincount(selected_node_coor[1], minlength=num_nodes)
        if node_selected_times.min() >= minimum_redundancy:
            break
        elif increase_rate and (counter % increase_interval) == 0:
            rate += 0.1 # enlarge the sampling rate 

    return selected, node_selected_times

def shortest_path_sampling(edge_index, node_coor_list, num_nodes, center_node_list, stride=2, 
                           minimum_redundancy=0, random_init=False):
    center_node_array = np.array(center_node_list)
    G = nx.from_edgelist(edge_index.numpy())
    G.add_nodes_from(range(num_nodes))
    if random_init: # here can also choose the one with highest degree
        index = np.random.choice(len(center_node_list))
        source_nodes = center_node_list[index]
    else:
        subgraph_size = torch.tensor([x.shape[1] for x in node_coor_list])
        index = subgraph_size.argmax().item()
        source_nodes = center_node_list[index]

    distance = np.ones(num_nodes)*1e10
    selected = []
    node_selected_times = torch.zeros(num_nodes)

    for i in tqdm(range(len(node_coor_list))):
        selected.append(index)
        node_selected_times[node_coor_list[index][1]] += 1
        for source in source_nodes:
            length_shortest_dict = nx.single_source_shortest_path_length(G, source)
            length_shortest = np.ones(num_nodes)*1e10
            length_shortest[list(length_shortest_dict.keys())] = list(length_shortest_dict.values())
            mask = length_shortest < distance
            distance[mask] = length_shortest[mask]
        
        if (distance.max() < stride) and (node_selected_times.min() >= minimum_redundancy): # stop criterion 
            break
        farthest = np.argmax(distance)
        if len(center_node_list[0]) > 1:
            candidates = center_node_array[:, center_node_array[:,0] == farthest]
            farthest_candidates = candidates[np.argmax(distance[candidates])]
            source_nodes = [farthest, farthest_candidates]
        else:
            source_nodes = [farthest]
        index = np.where(np.equal(center_node_array, source_nodes).all(1))[0].item()
    return selected, node_selected_times
 
# def min_set_cover_sampling(edge_index, subgraphs_nodes_mask, random_init=False, minimum_redundancy=2):

#     num_nodes = subgraphs_nodes_mask.size(0)
#     if random_init:
#         selected = np.random.choice(num_nodes) 
#     else:
#         selected = subgraphs_nodes_mask.sum(-1).argmax().item()

#     node_selected_times = torch.zeros(num_nodes)
#     selected_all = []

#     for i in range(num_nodes):
#         # selected_subgraphs[selected] = True
#         selected_all.append(selected)
#         node_selected_times[subgraphs_nodes_mask[selected]] += 1
#         if node_selected_times.min() >= minimum_redundancy: # stop criterion 
#             break
#         # calculate how many unused nodes in each subgraph (greedy set cover)
#         unused_nodes = ~ ((node_selected_times - node_selected_times.min()).bool())
#         num_unused_nodes = (subgraphs_nodes_mask & unused_nodes).sum(-1)
#         scores = num_unused_nodes
#         scores[selected_all] = 0
#         selected = np.argmax(scores).item()

#     return selected_all, node_selected_times