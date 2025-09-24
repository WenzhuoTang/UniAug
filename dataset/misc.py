import os
import math
import json
import random
import zipfile
import urllib.request
import numpy as np
import networkx as nx
import scipy.sparse as ssp

from tqdm import tqdm
from collections import defaultdict
from scipy.sparse.csgraph import shortest_path

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_dense_adj,
    to_undirected,
    add_self_loops,  # add_self_loops | add_remaininig_self_loops
    dense_to_sparse,
    negative_sampling, 
    train_test_split_edges,
)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))

def str2int(str):
    if 'K' in str:
        return int(str.split('K')[0]) * 1e3
    elif 'M' in str:
        return int(str.split('M')[0]) * 1e6
    elif 'B' in str:
        return int(str.split('B')[0]) * 1e9
    elif str.isdecimal():
        return float(str)
    else:
        return np.nan


# -------- split the nodes by labels --------
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_disassortative_splits(labels):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    num_classes = len(labels.unique())
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask, val_mask, test_mask

def multiple_random_splits(labels, num_rep=5):
    train_mask_list, val_mask_list, test_mask_list = [], [], []
    for _ in range(num_rep):
        train_mask, val_mask, test_mask = random_disassortative_splits(labels)
        train_mask_list.append(train_mask.unsqueeze(-1))
        val_mask_list.append(val_mask.unsqueeze(-1))
        test_mask_list.append(test_mask.unsqueeze(-1))
    
    train_mask = torch.cat(train_mask_list, dim=1)
    val_mask = torch.cat(val_mask_list, dim=1)
    test_mask = torch.cat(test_mask_list, dim=1)

    return train_mask, val_mask, test_mask


# -------- inject diffusion attributes --------
def get_diffusion_attributes(data):
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].long()
    row, col = torch.triu_indices(data.num_nodes, data.num_nodes, 1)
    data.full_edge_index = torch.stack([row, col])
    data.full_edge_attr = adj[data.full_edge_index[0], data.full_edge_index[1]]
    data.nodes_per_graph = data.num_nodes
    data.edges_per_graph = data.num_nodes * (data.num_nodes - 1) // 2

    return data

# -------- read benchmarking data for link prediction --------
# https://github.com/Juanhui28/HeaRT/
def read_link_prediction_data(dataset, dir_path, filename='samples.npy', setting='existing',
                              neg_mode='equal', return_type='data'):
    # setting in ['existing', 'heart']
    # neg_mode in ['equal', 'all']

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    for split in ['train', 'test', 'valid']:
        path = dir_path + '/{}/{}_pos.txt'.format(dataset, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            node_set.add(sub)
            node_set.add(obj)
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
            elif split == 'valid': 
                valid_pos.append((sub, obj))  
            elif split == 'test': 
                test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + dataset + ' is: ', num_nodes)

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    if setting == 'heart':
        with open(f'{dir_path}/{dataset}/heart_valid_{filename}', "rb") as f:
            valid_neg = np.load(f)
            # valid_neg = torch.from_numpy(valid_neg)
        with open(f'{dir_path}/{dataset}/heart_test_{filename}', "rb") as f:
            test_neg = np.load(f)
            # test_neg = torch.from_numpy(test_neg)

    elif setting == 'existing':
        for split in ['test', 'valid']:
            if neg_mode == 'equal':
                path = f'{dir_path}/{dataset}/{split}_neg.txt'
            elif neg_mode == 'all':
                path = f'{dir_path}/{dataset}/allneg/{split}_neg.txt'
            else:
                raise NotImplementedError(f'Unsupported neg_mode {neg_mode}')

            for line in open(path, 'r'):
                sub, obj = line.strip().split('\t')
                sub, obj = int(sub), int(obj)
                if split == 'valid': 
                    valid_neg.append((sub, obj))
                if split == 'test': 
                    test_neg.append((sub, obj))
    else:
        raise NotImplementedError(f'Unsupported setting {setting}')

    # adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
    train_pos_tensor = torch.tensor(train_pos)
    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)
    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)
    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(dir_path+ '/{}/{}'.format(dataset, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['x'] = feature_embeddings
    data['edge_index'] = edge_index.long()
    data['edge_weight'] = edge_weight

    if return_type == 'data':
        data['train_pos'] = train_pos_tensor.T
        data['train_val'] = train_val.T
        data['valid_pos'] = valid_pos.T
        data['valid_neg'] = valid_neg.T
        data['test_pos'] = test_pos.T
        data['test_neg'] = test_neg.T
        return Data(**data).coalesce()

    elif return_type == 'seal':
        split_edge = {'train': {}, 'valid': {}, 'test': {}}
        split_edge['train']['edge'] = edge_index
        split_edge['train']['edge_val'] = train_val.T
        split_edge['valid']['edge'] = valid_pos.T
        split_edge['valid']['edge_neg'] = valid_neg.T
        split_edge['test']['edge'] = test_pos.T
        split_edge['test']['edge_neg'] = test_neg.T
        return Data(**data).coalesce(), split_edge

    else:
        raise NotImplementedError(f'Unsupported return type {return_type}')

# -------- utils for debatch PYG data --------
def batched_to_list(bdata, keys_to_keep=None):
    ptr = bdata.ptr
    data_list = []
    for i in range(len(ptr) - 1):
        lower, upper = ptr[i], ptr[i + 1]
        data_dict = {}

        keys = keys_to_keep if keys_to_keep is not None else bdata.keys()
        for k in keys:
            if k in bdata.keys():
                # if k == 'edge_index':
                #     data_dict[k] = bdata[k][
                #         :, torch.logical_and(bdata.edge_index >= lower, bdata.edge_index < upper).all(0)
                #     ]

                if bdata.is_node_attr(k):
                    data_dict[k] = bdata[k][lower:upper]

                elif bdata.is_edge_attr(k):
                    try:
                        data_dict[k] = bdata[k][
                            torch.logical_and(bdata.edge_index >= lower, bdata.edge_index < upper).all(0)
                        ]
                    except:
                        data_dict[k] = bdata[k][
                            :, torch.logical_and(bdata.edge_index >= lower, bdata.edge_index < upper).all(0)
                        ]

                else:
                    try:
                        if bdata[k].shape[0] == len(ptr) - 1:  # graph attributes
                            data_dict[k] = bdata[k][i].unsqueeze(0)
                    except:
                        continue

        if data_dict['edge_index'].shape[1] > 0 and data_dict['edge_index'].min() > 0:
            # manually map edge_index
            mapping = dict(zip(range(lower, upper), range(upper - lower)))
            data_dict['edge_index'].apply_(lambda x: mapping.__getitem__(x))

        data_list.append(Data(**data_dict).coalesce())

    return data_list


# -------- read benchmarking data for heterophilic graphs --------
def load_data_new(data_name, root):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)

    if dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:

        adj_url = f'https://raw.githubusercontent.com/RecklessRonan/GloGNN/master/small-scale/new_data/{data_name}/out1_graph_edges.txt'
        feat_url = f'https://raw.githubusercontent.com/RecklessRonan/GloGNN/master/small-scale/new_data/{data_name}/out1_node_feature_label.txt'
        download_file(adj_url, f'{root}/{data_name}/raw/deezer-europe-splits.npy')
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        # print(sorted(graph_dict))
        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print('adj', adj.shape)
    # print('features', features.shape)
    # print('labels', labels.shape)
    # print('idx_train', idx_train.shape)
    # print('idx_val', idx_val.shape)
    # print('idx_test', idx_test.shape)
    return adj, features, labels, idx_train, idx_val, idx_test


# -------- utils for downloading and unzipping --------
def delete_file(path):
    if not os.path.exists(path):
        print("File does not exist")
    else:
        print(f"Deleting {path}")
        os.remove(path)

def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        u = urllib.request.urlopen(url)
        f = open(path, "wb")
        meta = u.info()
        file_size = int(meta.get("Content-Length", 0))
        print(f"Downloading: {path} Bytes: {file_size:,}")

        file_size_dl = 0
        block_sz = 8192
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                bar.update(len(buffer))
        f.close()
        u.close()
        return True
    else:
        print("File already downloaded")
        return False


def unzip_file(path, directory_to_extract_to):
    if not os.path.exists(path):
        print("File does not exist")
    else:
        print(f"Unzipping {path}")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        delete_file(path)


def download_unzip(url, path):
    zip_filepath = f"{path}.zip"
    download_file(url, zip_filepath)
    unzip_file(zip_filepath, path)


# -------- utils for graph property prediction datasets --------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Graph(nx.Graph):
    def __init__(self, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.laplacians = None
        self.v_plus = None

    def get_edge_index(self):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False):
        features = []
        for node, node_attrs in self.nodes(data=True):
            data = []

            if node_attrs["label"] is not None:
                data.extend(node_attrs["label"])

            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_node_degree:
                data.extend([self.degree(node)])

            if use_one:
                data.extend([1])
            
            features.append(data)
        
        return torch.Tensor(features)

    def get_target(self, classification=True):
        if classification:
            return torch.LongTensor([self.target])

        return torch.Tensor([self.target])

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def parse_tu_data(name, raw_dir):
    # setup paths
    indicator_path = raw_dir / name / f'{name}_graph_indicator.txt'
    edges_path = raw_dir / name / f'{name}_A.txt'
    graph_labels_path = raw_dir / name / f'{name}_graph_labels.txt'
    node_labels_path = raw_dir / name / f'{name}_node_labels.txt'
    edge_labels_path = raw_dir / name / f'{name}_edge_labels.txt'
    node_attrs_path = raw_dir / name / f'{name}_node_attributes.txt'
    edge_attrs_path = raw_dir / name / f'{name}_edge_attributes.txt'

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1,-1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)

    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            edge_indicator.append(edge)

            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            graph_id = indicator[edge[0]]

            graph_edges[graph_id].append(edge)

    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                unique_node_labels.add(node_label)
                graph_id = indicator[i]
                node_labels[graph_id].append(node_label)

    if edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                edge_label = int(line)
                unique_edge_labels.add(edge_label)
                graph_id = indicator[edge_indicator[i][0]]
                edge_labels[graph_id].append(edge_label)

    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                node_attr = np.array([float(n) for n in nums])
                graph_id = indicator[i]
                node_attrs[graph_id].append(node_attr)

    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                edge_attr = np.array([float(n) for n in nums])
                graph_id = indicator[edge_indicator[i][0]]
                edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            target = int(line)
            if target == -1:
                graph_labels.append(0)
            else:
                graph_labels.append(target)

        if min(graph_labels) == 1:  # Shift by one to the left. Apparently this is necessary for multiclass tasks.
            graph_labels = [l - 1 for l in graph_labels]

    num_node_labels = max(unique_node_labels) if unique_node_labels != set() else 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:  # some datasets e.g. PROTEINS have labels with value 0
        num_node_labels += 1

    num_edge_labels = max(unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs
    }, num_node_labels, num_edge_labels


def create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels):
    nodes = graph_data["graph_nodes"]
    edges = graph_data["graph_edges"]

    G = Graph(target=target)

    for i, node in enumerate(nodes):
        label, attrs = None, None

        if graph_data["node_labels"] != []:
            label = one_hot(graph_data["node_labels"][i], num_node_labels)

        if graph_data["node_attrs"] != []:
            attrs = graph_data["node_attrs"][i]

        G.add_node(node, label=label, attrs=attrs)

    for i, edge in enumerate(edges):
        n1, n2 = edge
        label, attrs = None, None

        if graph_data["edge_labels"] != []:
            label = one_hot(graph_data["edge_labels"][i], num_edge_labels)
        if graph_data["edge_attrs"] != []:
            attrs = graph_data["edge_attrs"][i]

        G.add_edge(n1, n2, label=label, attrs=attrs)

    return G


# -------- utils for SEAL dataset --------
def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    center_flag = torch.zeros(len(node_features), dtype=bool)
    center_flag[:2] = True  # first 2 nodes
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes, center_flag=center_flag)
    return data

 
def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None, 
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.T.tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x, y=y, 
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg']
        # subsample for pos_edge
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge


# -------- utils for networkx to pyg --------
def preprocess(g, degree=False):
    if isinstance(g, nx.Graph):
        pyg_data = torch_geometric.utils.from_networkx(g)
    elif isinstance(g, torch_geometric.data.Data):
        pyg_data = g
    else:
        raise NotImplementedError()

    pyg_data.edge_index = to_undirected(
        pyg_data.edge_index, num_nodes=pyg_data.num_nodes, reduce='mean'
    )
    adj = to_dense_adj(pyg_data.edge_index)[0].long()

    row, col = torch.triu_indices(pyg_data.num_nodes, pyg_data.num_nodes,1)
    pyg_data.full_edge_index = torch.stack([row, col])
    pyg_data.full_edge_attr = adj[pyg_data.full_edge_index[0], pyg_data.full_edge_index[1]]
    pyg_data.nodes_per_graph = pyg_data.num_nodes
    pyg_data.edges_per_graph = pyg_data.num_nodes * (pyg_data.num_nodes - 1) // 2

    if degree:
        pyg_data.degree = torch_geometric.utils.degree(pyg_data.edge_index[0]).long() 

    return pyg_data

# -------- Archived --------

# from utils.mol_utils import bond_to_feature_vector as bond_to_feature_vector_non_santize
# from utils.mol_utils import atom_to_feature_vector as atom_to_feature_vector_non_santize

# from rdkit import Chem
# from rdkit.Chem import AllChem


# def smiles2graph(smiles_string, sanitize=True):
#     """
#     Converts SMILES string to graph Data object
#     :input: SMILES string (str)
#     :return: graph object
#     """
#     try:
#         mol = Chem.MolFromSmiles(smiles_string, sanitize=sanitize)
#         # atoms
#         atom_features_list = []
#         atom_label = []
#         # print('smiles_string', smiles_string)
#         # print('mol', Chem.MolToSmiles(mol), 'vs smiles_string', smiles_string)
#         for atom in mol.GetAtoms():
#             if sanitize:
#                 atom_features_list.append(atom_to_feature_vector(atom))
#             else:
#                 atom_features_list.append(atom_to_feature_vector_non_santize(atom))

#             atom_label.append(atom.GetSymbol())

#         x = np.array(atom_features_list, dtype = np.int64)
#         atom_label = np.array(atom_label, dtype = np.str)

#         # bonds
#         num_bond_features = 3  # bond type, bond stereo, is_conjugated
#         if len(mol.GetBonds()) > 0: # mol has bonds
#             edges_list = []
#             edge_features_list = []
#             for bond in mol.GetBonds():
#                 i = bond.GetBeginAtomIdx()
#                 j = bond.GetEndAtomIdx()

#                 # edge_feature = bond_to_feature_vector(bond)
#                 if sanitize:
#                     edge_feature = bond_to_feature_vector(bond)
#                 else:
#                     edge_feature = bond_to_feature_vector_non_santize(bond)
#                 # add edges in both directions
#                 edges_list.append((i, j))
#                 edge_features_list.append(edge_feature)
#                 edges_list.append((j, i))
#                 edge_features_list.append(edge_feature)

#             # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
#             edge_index = np.array(edges_list, dtype = np.int64).T

#             # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#             edge_attr = np.array(edge_features_list, dtype = np.int64)

#         else:   # mol has no bonds
#             edge_index = np.empty((2, 0), dtype = np.int64)
#             edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

#         graph = dict()
#         graph['edge_index'] = edge_index
#         graph['edge_feat'] = edge_attr
#         graph['node_feat'] = x
#         graph['num_nodes'] = len(x)
        
#         return graph 

#     except:
#         return None

# def labeled2graphs(raw_dir):
#     '''
#         - raw_dir: the position where property csv stored,  
#     '''
#     path_suffix = pathlib.Path(raw_dir).suffix
#     if path_suffix == '.csv':
#         df_full = pd.read_csv(raw_dir, engine='python')
#         df_full.set_index('SMILES', inplace=True)
#         print(df_full[:5])
#     else:
#         raise ValueError("Support only csv.")
#     graph_list = []
#     for smiles_idx in tqdm(df_full.index[:]):
#         graph_dict = smiles2graph(smiles_idx)
#         props = df_full.loc[smiles_idx]
#         for (name,value) in props.iteritems():
#             graph_dict[name] = np.array([[value]])
#         graph_list.append(graph_dict)
#     return graph_list

# def read_graph_list(raw_dir, property_name=None, drop_property=False, process_labeled=False):
#     print('raw_dir', raw_dir)
#     assert process_labeled
#     graph_list = labeled2graphs(raw_dir)

#     pyg_graph_list = []
#     print('Converting graphs into PyG objects...')
#     for graph in graph_list:
#         g = Data()
#         g.__num_nodes__ = graph['num_nodes']
#         g.edge_index = torch.from_numpy(graph['edge_index'])
#         del graph['num_nodes']
#         del graph['edge_index']
#         if process_labeled:
#             g.y = torch.from_numpy(graph[property_name.split('-')[1]])
#             del graph[property_name.split('-')[1]]

#         if graph['edge_feat'] is not None:
#             g.edge_attr = torch.from_numpy(graph['edge_feat'])
#             del graph['edge_feat']

#         if graph['node_feat'] is not None:
#             g.x = torch.from_numpy(graph['node_feat'])
#             del graph['node_feat']

#         addition_prop = copy.deepcopy(graph)
#         for key in addition_prop.keys():
#             g[key] = torch.tensor(graph[key])
#             del graph[key]

#         pyg_graph_list.append(g)

#     return pyg_graph_list