# from collections.abc import Iterable

import os
import os.path as osp
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm

import torch
from torch_sparse import SparseTensor, coalesce

import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    subgraph,
    to_undirected,
    remove_isolated_nodes,
    add_remaining_self_loops,
)
from torch_geometric.transforms import (
    NormalizeFeatures,
    SVDFeatureReduction, 
    ToSparseTensor,
)

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset


from dataset.subgraph import extract_subgraphs, extract_segments
from dataset.feature import get_features, aggregate_edge_attr
from dataset.loader import (
    MultiEpochsPYGDataLoader,
    graphs_to_feat_and_adj_dataloader
)
from dataset.misc import (
    download_unzip,
    k_hop_subgraph,
    get_pos_neg_edges,
    construct_pyg_graph,
    read_link_prediction_data,
    extract_enclosing_subgraphs,
)


FEATURE_CHOICE = ['node2vec', 'cn', 'aa', 'ra', 'ppr', 'katz']
DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'ppr']


class InducedDataset(InMemoryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(None, None, None)

    # TODO: implement segment training
    # https://github.com/kaidic/GST/blob/main/graphgps/loader/dataset/malnet_large.py
    # def segment_and_fill(self):
    #     N, E = graph.num_nodes, graph.num_edges
    #     adj = SparseTensor(
    #         row=graph.edge_index[0], col=graph.edge_index[1],
    #         value=torch.arange(E, device=graph.edge_index.device),
    #         sparse_sizes=(N, N))
    #     adj = adj.to_symmetric()
    #     num_partition = N // self.thres + 1
    #     adj, partptr, perm = adj.partition(num_partition, False)
    
    def get_dataloader(self, type='pyg', split=None, batch_size=64, shuffle=False, **kwargs):
        if split is not None:
            split = [split] if not isinstance(split, list) else split
            idx = []
            for s in split:
                assert s in list(self.splits), f'{s} not in {list(self.splits)}'
                idx.extend(self.splits[s])
            dataset = self[idx]
        else:
            dataset = self

        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(dataset, batch_size, shuffle)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(dataset, batch_size, shuffle)
        else:
            raise NotImplementedError(f'Unsupported type {type}')

    def prepare_data(self):
        pass


class UnlabeledInducedDataset(InducedDataset):
    NETWORK_REPOSITORY_LINKS = {
        'ca-IMDB': 'https://nrvis.com/download/data/ca/ca-IMDB.zip',
        'ca-CondMat': 'https://nrvis.com/download/data/ca/ca-CondMat.zip',
        'ca-AstroPh': 'https://nrvis.com/download/data/ca/ca-AstroPh.zip',
        'ca-HepPh': 'https://nrvis.com/download/data/ca/ca-HepPh.zip',
        'bio-CE-CX': 'https://nrvis.com/download/data/bio/bio-CE-CX.zip',
        'econ-poli-large': 'https://nrvis.com/download/data/econ/econ-poli-large.zip',
        'email-EU': 'https://nrvis.com/download/data/email/email-EU.zip',
        'rec-movielens-tag-movies-10m': 'https://nrvis.com/download/data/rec/rec-movielens-tag-movies-10m.zip',
        'soc-epinions': 'https://nrvis.com/download/data/soc/soc-epinions.zip',
        'soc-anybeat': 'https://nrvis.com/download/data/soc/soc-anybeat.zip'
    }
    DATASET_DICT = {
        'network_repository': [
            'ca-CondMat', 'ca-HepPh', 'bio-CE-CX', 'email-EU', 'rec-movielens-tag-movies-10m', 'soc-epinions'
        ],  # bad ones: 'ca-AstroPh', 'econ-poli-large', 
        'planetoid': ['citeseer', 'pubmed', 'cora'],
        'pyg': ['QM9', 'ZINC'],
    }

    def __init__(
            self, data_name_list=['ca-CondMat'], dir_path='./data/misc', subgraph_type='ego', 
            num_hops=2, walk_length=10, repeat=5, max_node_num=100, sampling_mode=None, 
            random_init=False, minimum_redundancy=1, shortest_path_mode_stride=5,
            random_mode_sampling_rate=0.5, feature_principle='full', feature_types=None, 
            embedding_dim=64, walk_length_node2vec=20, context_size=20, walks_per_node=1,
            p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, beta_katz=0.005, 
            path_len=3, remove=False, aggr='add', aggr_repeat=2, fill_zeros=True, 
            add_self_loop=False, force_process=False, save=True, **kwargs
        ):
        super().__init__()

        if feature_types is None:
            feature_types = DEFAULT_FEATURE_TYPES
        self.prepare_data(
            data_name_list, dir_path, subgraph_type, num_hops, walk_length, repeat, max_node_num, 
            sampling_mode, random_init, minimum_redundancy, shortest_path_mode_stride, 
            random_mode_sampling_rate, feature_principle, feature_types, embedding_dim, walk_length_node2vec, 
            context_size, walks_per_node, p_node2vec, q_node2vec, num_negative_samples, p_ppr, beta_katz, 
            path_len, remove, aggr, aggr_repeat, fill_zeros, add_self_loop, force_process, save
        )
    
    def check_data_name(self, data_name_list):
        # sanity check of data_name_list
        if not isinstance(data_name_list, list):
            if data_name_list in list(self.DATASET_DICT):
                data_name_list = self.DATASET_DICT[data_name_list]
            elif data_name_list == 'all':
                import itertools
                data_name_list = list(itertools.chain.from_iterable(self.DATASET_DICT.values()))
            else:
                data_name_list = [data_name_list]
        else:
            temp_list = []
            for d in data_name_list:
                if d in self.DATASET_DICT.keys():
                    temp_list.extend(self.DATASET_DICT[d])
                else:
                    temp_list.append(d)
            data_name_list = temp_list
        return data_name_list

    def get_data(self, data_name='ca-CondMat', dir_path='./data/misc', remove_isolated=True, add_self_loop=False):
        if data_name in self.DATASET_DICT['network_repository']:
            if not osp.exists(f'{dir_path}/{data_name}'):
                url = self.NETWORK_REPOSITORY_LINKS[data_name]
                download_unzip(url, f'{dir_path}/{data_name}')
            files = os.listdir(f'{dir_path}/{data_name}')
            data_file = [x for x in files if x.startswith(data_name)][0]
            try:
                from scipy.io import mmread
                
                edges_coo = mmread(f'{dir_path}/{data_name}/{data_file}')
                edge_index = torch.stack((torch.from_numpy(edges_coo.col), torch.from_numpy(edges_coo.row))).long()
                edge_attr = torch.from_numpy(edges_coo.data)
            except:
                import pandas as pd

                edges_df = pd.read_csv(f'{dir_path}/{data_name}/{data_file}', header=None, sep=' ')
                if data_file.endswith('mtx'):
                    # remove irrelavent rows
                    counter = 0
                    while True:
                        try:
                            int(edges_df.iloc[counter, 0])
                            break
                        except:
                            counter += 1
                    edges_df = edges_df.iloc[counter:]
                    edges_df.iloc[:,0] = edges_df.iloc[:,0].astype(int)
                    edges_df.iloc[:,1] = edges_df.iloc[:,1].astype(int)

                edge_index = torch.stack((
                    torch.tensor(edges_df.iloc[:,0].values, dtype=torch.long), 
                    torch.tensor(edges_df.iloc[:,1].values, dtype=torch.long)
                ))
                if edges_df.shape[1] > 2 and not any(edges_df.iloc[:,2].isna()):
                    edge_attr = torch.tensor(edges_df.iloc[:,2].values.astype(float))
                else:
                    edge_attr = None

            # modification
            if remove_isolated:
                edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr)
            if add_self_loop:
                edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)

            data = Data(edge_index=edge_index, edge_attr=edge_attr)

            return data, 'data'

        elif data_name in self.DATASET_DICT['planetoid']:
            dataset = Planetoid(name=data_name, root=dir_path)
            return dataset._data, 'data'

        elif data_name in self.DATASET_DICT['pyg']:
            dataset_class = getattr(torch_geometric.datasets, data_name)
            dataset = dataset_class(root=dir_path)
            return dataset, 'dataset'

        else:
            raise NotImplementedError(f'Unsuppoted data_name {data_name}')

    def prepare_data(
            self, data_name_list=['ca-CondMat'], dir_path='./data/misc', subgraph_type='ego', 
            num_hops=2, walk_length=10, repeat=5, max_node_num=100, sampling_mode=None, 
            random_init=False, minimum_redundancy=3, shortest_path_mode_stride=2, 
            random_mode_sampling_rate=0.5, feature_principle='sub', feature_types=['all'], 
            embedding_dim=64, walk_length_node2vec=20, context_size=20, walks_per_node=1,
            p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, beta_katz=0.005, 
            path_len=3, remove=False, aggr='add', aggr_repeat=2, fill_zeros=False, 
            add_self_loop=False, force_process=False, save=True
        ):
        if subgraph_type == 'ego':
            prefix = f'{num_hops}_hop'
        elif subgraph_type == 'rw':
            prefix = f'length_{walk_length}-repeat_{repeat}'
        else:
            raise NotImplementedError(f'Unsupported subgraph_type {subgraph_type}')
        
        if sampling_mode is not None:
            if sampling_mode == 'random':
                postfix = f'-{sampling_mode}-min_red{minimum_redundancy}'
            elif sampling_mode == 'shortest_path':
                postfix = f'-{sampling_mode}-min_red{minimum_redundancy}-stride{shortest_path_mode_stride}'
            else:
                raise NotImplementedError(f'Unsupported sampling_mode {sampling_mode}')
        else:
            postfix = ''

        data_name = '_'.join(sorted(data_name_list))
        feat_name = '_'.join(sorted(feature_types))
        save_name = f'-max_{max_node_num}-feat_{feat_name}'
        fname = f'{dir_path}/{data_name}/{prefix}{save_name}{postfix}-subgraphs.pt'

        # frame = inspect.currentframe()
        # _, _, _, values = inspect.getargvalues(frame)
        # configs = {args[i]: values[i] for i in range(len(args))}
        # keys_to_ignore = ['self', 'data_name_list', 'dir_path', 'force_process', 'save']
        # for k in keys_to_ignore:
        #     configs.pop(k)

        data_name_list = self.check_data_name(data_name_list)
        if force_process or not osp.exists(fname):
            os.makedirs(f'{dir_path}/{data_name}', exist_ok=True)
            subgraphs = {}
            for d in data_name_list:
                # assert d not in self.DATASET_DICT['pyg'], f'Cannot get induced graphs of dataset {d}'
                print(f'Dataset: {d}')
                data_fname = f'{dir_path}/{d}/{prefix}{save_name}{postfix}-subgraphs.pt'
                if force_process or not osp.exists(data_fname):
                    data, type = self.get_data(d, dir_path)
                    if type == 'dataset':
                        # TODO: add node2vec
                        subgraphs[d] = list(data)
                    else:
                        if feature_principle == 'full':
                            replace_feat = False
                            n2v_prefix = f'-dim_{embedding_dim}' if 'node2vec' in feature_types else None
                            save_path = f'{dir_path}/{d}/{feat_name}{n2v_prefix}-embdedding.pt'
                            if force_process or not osp.exists(save_path):
                                feat_dict = get_features(
                                    data.edge_index, data.edge_attr, None, feature_types, embedding_dim, 
                                    walk_length_node2vec, context_size, walks_per_node, p_node2vec, 
                                    q_node2vec, num_negative_samples, p_ppr, beta_katz, path_len, remove, 
                                    print_loss=True, return_dict=True,
                                )
                                print(f'Saving embdeddings to {save_path}')
                                torch.save(feat_dict, save_path)
                            else:
                                print(f'Loading embdeddings from {save_path}')
                                feat_dict = torch.load(save_path)

                            node_feat = feat_dict.pop('node2vec')
                            edge_attr = torch.cat(list(feat_dict.values()), dim=1) if len(feat_dict) > 0 else None
                            if edge_attr is not None:
                                data.x = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                            repeat=aggr_repeat, aggr=aggr)
                            else:
                                data.x = node_feat

                            # sanity check for data.x
                            data.x = torch.nan_to_num(data.x)

                        elif feature_principle == 'sub':
                            replace_feat = True
                        else:
                            raise NotImplementedError(f'Unknown feature_principle {feature_principle}')

                        nodes_list = [[x] for x in range(data.num_nodes)]
                        y = None
                        subgraphs[d] = extract_subgraphs(
                            data, y, nodes_list, subgraph_type, num_hops, walk_length, repeat, max_node_num,
                            sampling_mode, random_init, minimum_redundancy, shortest_path_mode_stride, 
                            random_mode_sampling_rate, fill_zeros=fill_zeros, add_self_loop=add_self_loop
                        )
                else:
                    print(f'Loading subgraphs from {data_fname}')
                    subgraphs[d] = torch.load(data_fname)[d]

            if save:
                print(f'Saving subgraphs to {fname}')
                torch.save(subgraphs, fname)

        else:
            print(f'Loading subgraphs from {fname}')
            subgraphs = torch.load(fname)
            # subgraphs.pop('configs')

        self.data_list = []
        self.splits = {}
        split_names = sorted(subgraphs)
        counter = 0
        for i in range(len(split_names)):
            temp_subgraphs = subgraphs[split_names[i]]
            self.splits[split_names[i]] = list(range(counter, counter + len(temp_subgraphs)))
            for g in temp_subgraphs:
                # sanity check for g.x
                g.x = torch.nan_to_num(g.x)
                if g.edge_attr is None:
                    g.edge_attr = torch.ones(g.num_edges, dtype=float)
            self.data_list.extend(temp_subgraphs)
            counter += len(temp_subgraphs)

        self.data, self.slices = self.collate(self.data_list)


class DownstreamInducedDataset(InducedDataset):
    DIR_NAME = {
        'LP': 'link_prediction',
        'NC': 'node_classification',
        'both': 'both',
    }
    LP_SPLITS = ['train_pos', 'valid_pos', 'valid_neg', 'test_pos', 'test_neg']
    MAX_NODE_NUM = {
        'LP':{
            'cora': {1: 162, 2: 458},
            'pubmed': {1: 244, 2: 2409},
            'citeseer': {1: 105, 2: 282},
            'all': {1: 244, 2: 2409},
        },
        'NC':{
            'cora': {1: 79, 2: 240},
            'pubmed': {1: 74, 2: 729},
            'citeseer': {1: 52, 2: 178},
            'all': {1: 79, 2: 729},
        }
    }
    FEAT_DIM = {'citeseer':3703, 'pubmed': 500, 'cora': 1433}
    ALL_DATASETS = ['citeseer', 'pubmed', 'cora']

    def __init__(self, task='LP', data_name='cora', dir_path='./data', filename='samples.npy',
                 out_channels=None, subgraph_type='ego', num_hops=2, walk_length=10, repeat=5, 
                 max_node_num=100, sampling_mode=None, random_init=False, minimum_redundancy=1, 
                 shortest_path_mode_stride=5, random_mode_sampling_rate=0.5, stru_feat_principle='concat',
                 feature_types=None, embedding_dim=64, walk_length_node2vec=20, context_size=20, 
                 walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, 
                 beta_katz=0.005, path_len=3, remove=False, aggr='add', aggr_repeat=2, fill_zeros=True, 
                 add_self_loop=False, add_subgraph_self_loop=False, force_process=False, save=True,
                 rank_nodes='feat_corr' ,**kwargs):

        super().__init__()
        assert task in ['LP','NC', 'both']
        self.task = task
        self.data_name = data_name
        self.num_hops = num_hops

        self.root_dir = dir_path
        dir_path = f'{dir_path}/{self.DIR_NAME[task]}'
        self.dir_path = dir_path

        if feature_types is None and stru_feat_principle is not None:
            feature_types = DEFAULT_FEATURE_TYPES
        # if max_node_num is None:
        #     max_node_num = self.MAX_NODE_NUM[task][data_name][num_hops]
        self.prepare_data(
            task, data_name, dir_path, filename, out_channels, subgraph_type, num_hops, walk_length,
            repeat, max_node_num, sampling_mode, random_init, minimum_redundancy, shortest_path_mode_stride, 
            random_mode_sampling_rate, stru_feat_principle, feature_types, embedding_dim, 
            walk_length_node2vec, context_size, walks_per_node, p_node2vec, q_node2vec, num_negative_samples, 
            p_ppr, beta_katz, path_len, remove, aggr, aggr_repeat, fill_zeros, add_self_loop, add_subgraph_self_loop,
            force_process, save, rank_nodes
        )

    def get_data(self, task='LP', data_name='cora', dir_path='./data', filename='samples.npy', 
                 out_channels=None, setting='existing', neg_mode='equal'):
        if task == 'LP':
            data = read_link_prediction_data(data_name, dir_path, filename, setting, neg_mode)

        elif task == 'NC':
            if data_name in ['citeseer', 'pubmed', 'cora']:
                dataset = Planetoid(name=data_name, root=dir_path)
                data = dataset.get(0)
                data.splits = {
                    'train': torch.where(data.train_mask)[0],
                    'valid': torch.where(data.val_mask)[0],
                    'test': torch.where(data.test_mask)[0],
                }
            else:
                dataset = PygNodePropPredDataset(name=data_name, root=dir_path)
                data = dataset.get(0)
                data.splits = dataset.get_idx_split()

        # feature dimension reduction with SVD
        if out_channels is not None:
            feature_reduce = SVDFeatureReduction(out_channels=out_channels)
            data = feature_reduce(data)

        return data
    
    def prepare_data(self, task='LP', data_name='cora', dir_path='./data', filename='samples.npy',
                     out_channels=None, subgraph_type='ego', num_hops=2, walk_length=10, repeat=5, 
                     max_node_num=100, sampling_mode=None, random_init=False, minimum_redundancy=1, 
                     shortest_path_mode_stride=5, random_mode_sampling_rate=0.5, stru_feat_principle='concat', 
                     feature_types=None, embedding_dim=64, walk_length_node2vec=20, context_size=20, 
                     walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, 
                     beta_katz=0.005, path_len=3, remove=False, aggr='add', aggr_repeat=2, fill_zeros=True, 
                     add_self_loop=False, add_subgraph_self_loop=False, force_process=False, save=True, 
                     rank_nodes='feat_corr'):
            
        if subgraph_type == 'ego':
            prefix = f'{num_hops}_hop'
        elif subgraph_type == 'rw':
            prefix = f'length_{walk_length}-repeat_{repeat}'
        else:
            raise NotImplementedError(f'Unsupported subgraph_type {subgraph_type}')
        
        if sampling_mode is not None:
            if sampling_mode == 'random':
                postfix = f'-{sampling_mode}-min_red{minimum_redundancy}'
            elif sampling_mode == 'shortest_path':
                postfix = f'-{sampling_mode}-min_red{minimum_redundancy}-stride{shortest_path_mode_stride}'
            else:
                raise NotImplementedError(f'Unsupported sampling_mode {sampling_mode}')
        else:
            postfix = ''

        feat_name = '_'.join(sorted(feature_types)) if feature_types is not None else 'none'
        save_name = f'-max_{max_node_num}-feat_{feat_name}-subsample_{rank_nodes}'
        fname = f'{dir_path}/{data_name}/{prefix}{save_name}{postfix}-subgraphs.pt'
        if force_process or not osp.exists(fname):
            if not osp.exists(f'{dir_path}/{data_name}'):
                os.mkdir(f'{dir_path}/{data_name}')

            subgraphs = {}
            data = self.get_data(task, data_name, dir_path, filename, out_channels)
            if add_self_loop:
                data.edge_index, data.edge_attr = add_remaining_self_loops(
                    data.edge_index, data.edge_attr, num_nodes=data.num_nodes
                )

            if stru_feat_principle is not None and feature_types is not None:  # calculate structure feature
                n2v_prefix = f'-dim_{embedding_dim}' if 'node2vec' in feature_types else None
                save_path = f'{dir_path}/{data_name}/{feat_name}{n2v_prefix}-embdedding.pt'
                if force_process or not osp.exists(save_path):
                    feat_dict = get_features(
                        data.edge_index, data.edge_attr, data.num_nodes, feature_types, embedding_dim, 
                        walk_length_node2vec, context_size, walks_per_node, p_node2vec, 
                        q_node2vec, num_negative_samples, p_ppr, beta_katz, path_len, remove, 
                        print_loss=True, return_dict=True,
                    )
                    print(f'Saving embdeddings to {save_path}')
                    torch.save(feat_dict, save_path)
                else:
                    print(f'Loading embdeddings from {save_path}')
                    feat_dict = torch.load(save_path)

                node_feat = feat_dict.pop('node2vec')
                edge_attr = torch.cat(list(feat_dict.values()), dim=1) if len(feat_dict) > 0 else None
                if edge_attr is not None:
                    structure_feat = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                         repeat=aggr_repeat, aggr=aggr)
                else:
                    structure_feat = node_feat

                # sanity check for structure feature
                structure_feat = torch.nan_to_num(structure_feat)
                if stru_feat_principle == 'concat':
                    data.x = torch.concat([data.x, structure_feat], dim=1)
                elif stru_feat_principle == 'replace':
                    data.x = structure_feat
                else:
                    raise NotImplementedError(f'Unsupported structure feature principle {stru_feat_principle}')

            split_names = self.LP_SPLITS if task == 'LP' else list(data.splits)
            for s in split_names:
                remove_center_edges = False
                if task == 'LP':
                    # only keep the upper triangle for the undirected graphs
                    split_edge_index = data[s].detach().cpu().clone()
                    mask = split_edge_index[0] < split_edge_index[1]
                    split_edge_index[0] = split_edge_index[0][mask]
                    split_edge_index[1] = split_edge_index[1][mask]

                    nodes_list = split_edge_index.T.tolist()
                    y = torch.tensor([1 if 'pos' in s else 0]).long().repeat(len(nodes_list))
                    if 'train' not in s:
                        remove_center_edges = True

                elif task == 'NC':
                    nodes_list = [[x.item()] for x in data.splits[s]]
                    y = data.y[data.splits[s]].long()

                _sampling_mode = sampling_mode if 'train' in s else None
                subgraphs[f'{data_name}-{task}-{s}'] = extract_subgraphs(
                    data, y, nodes_list, subgraph_type, num_hops, walk_length, repeat, max_node_num,
                    _sampling_mode, random_init, minimum_redundancy, shortest_path_mode_stride, 
                    random_mode_sampling_rate, fill_zeros=fill_zeros, add_self_loop=add_subgraph_self_loop,
                    rank_nodes=rank_nodes, remove_center_edges=remove_center_edges,
                )
            if save:
                print(f'Saving subgraphs to {fname}')
                torch.save(subgraphs, fname)
        else:
            print(f'Loading subgraphs from {fname}')
            subgraphs = torch.load(fname)

        self.data_list = []
        self.splits = {}
        split_names = sorted(subgraphs)
        counter = 0
        for i in range(len(split_names)):
            temp_subgraphs = subgraphs[split_names[i]]
            self.splits[split_names[i]] = list(range(counter, counter + len(temp_subgraphs)))
            self.data_list.extend(temp_subgraphs)
            counter += len(temp_subgraphs)

        self.data, self.slices = self.collate(self.data_list)
        if task == 'LP':
            self._data.y = self._data.y.view(-1, 1)
        elif task == 'NC':
            self._data.y = torch.nn.functional.one_hot(self._data.y)
        else:
            raise NotImplementedError(f'Unsupoprted task {task}')


class SEALDataset(InMemoryDataset):
    DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'ppr']

    def __init__(self, data_name='cora', dir_path='./data', num_hops=3, percent=100, use_coalesce=False, 
                 node_label='drnl', ratio_per_hop=1.0, max_nodes_per_hop=None, directed=False, # SEAL kwargs
                 stru_feat_principle=None, feature_types=None, embedding_dim=64, walk_length_node2vec=20, 
                 context_size=20, walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, 
                 p_ppr=0.85, beta_katz=0.005, path_len=3, remove=False, aggr='add', aggr_repeat=2, **kwargs):
        self.data_name = data_name
        self.dir_path = f'{dir_path}/link_prediction'
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed

        if feature_types is None and stru_feat_principle is not None:
            feature_types = DEFAULT_FEATURE_TYPES
        self.feat_name = '_'.join(sorted(feature_types)) if feature_types is not None else 'none'
        self.stru_feat_principle = stru_feat_principle
        self.feature_types = feature_types
        self.feat_config = {
            'embedding_dim': embedding_dim, 'walk_length': walk_length_node2vec, 'context_size': context_size,
            'walks_per_node': walks_per_node, 'p_node2vec': p_node2vec, 'q_node2vec': q_node2vec,
            'num_negative_samples': num_negative_samples, 'p_ppr': p_ppr, 'beta_katz': beta_katz,
            'path_len': path_len, 'remove': remove,
        }
        self.aggr = aggr
        self.aggr_repeat = aggr_repeat
        root = f'{self.dir_path}/{self.data_name}'
        super(SEALDataset, self).__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')
        self.data, self.slices, self.splits = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        save_name = f'SEAL-{self.num_hops}_hop-{self.percent}_percent-feat_{self.feat_name}'
        if self.max_nodes_per_hop is not None:
            save_name += f'-max_{self.max_nodes_per_hop}_per_hop'
        save_name += '.pt'
        return [save_name]

    def process(self):
        data, split_edge = read_link_prediction_data(self.data_name, self.dir_path, return_type='seal')
        if self.use_coalesce:  # compress mutli-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight, 
                data.num_nodes, data.num_nodes)
        
        if self.stru_feat_principle is not None and self.feature_types is not None:  # calculate structure feature
            n2v_prefix = f"-dim_{self.feat_config['embedding_dim']}" if 'node2vec' in self.feature_types else None
            save_path = f'{self.dir_path}/{self.data_name}/{self.feat_name}{n2v_prefix}-embdedding.pt'
            if not osp.exists(save_path):
                feat_dict = get_features(
                    data.edge_index, data.edge_weight, data.num_nodes, self.feature_types,
                    print_loss=True, return_dict=True, **self.feat_config
                )
                print(f'Saving embdeddings to {save_path}')
                torch.save(feat_dict, save_path)
            else:
                print(f'Loading embdeddings from {save_path}')
                feat_dict = torch.load(save_path)

            node_feat = feat_dict.pop('node2vec')
            edge_attr = torch.cat(list(feat_dict.values()), dim=1) if len(feat_dict) > 0 else None
            if edge_attr is not None:
                structure_feat = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                     repeat=self.aggr_repeat, aggr=self.aggr)
            else:
                structure_feat = node_feat

            # sanity check for structure feature
            structure_feat = torch.nan_to_num(structure_feat)
            if self.stru_feat_principle == 'concat':
                data.x = torch.concat([data.x, structure_feat], dim=1)
            elif self.stru_feat_principle == 'replace':
                data.x = structure_feat
            else:
                raise NotImplementedError(f'Unsupported structure feature principle {self.stru_feat_principle}')

        # if 'edge_weight' in data:
        if hasattr(data, 'edge_weight')  and data.edge_weight != None:
            edge_weight = data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (data.edge_index[0], data.edge_index[1])), 
            shape=(data.num_nodes, data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None
        
        # Extract enclosing subgraphs for pos and neg edges
        split_names = sorted(split_edge)
        data_list = []
        splits = {}
        counter = 0
        for s in split_names:
            pos_edge, neg_edge = get_pos_neg_edges(s, split_edge, data.edge_index, 
                                                   data.num_nodes, self.percent)
            pos_list = extract_enclosing_subgraphs(
                pos_edge, A, data.x, 1, self.num_hops, self.node_label, 
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
            splits[f'{self.data_name}-LP-{s}_pos'] = list(range(counter, counter + len(pos_list)))
            data_list.extend(pos_list)
            counter += len(pos_list)

            neg_list = extract_enclosing_subgraphs(
                neg_edge, A, data.x, 0, self.num_hops, self.node_label, 
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
            splits[f'{self.data_name}-LP-{s}_neg'] = list(range(counter, counter + len(neg_list)))
            data_list.extend(neg_list)
            counter += len(neg_list)

        self.data, self.slices = self.collate(data_list)

        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self._data, self.slices, splits), self.processed_paths[0])

    def get_dataloader(self, type='pyg', split=None, batch_size=64, shuffle=False, **kwargs):
        if split is not None:
            split = [split] if not isinstance(split, list) else split
            idx = []
            for s in split:
                assert s in list(self.splits), f'{s} not in {list(self.splits)}'
                idx.extend(self.splits[s])
            dataset = self[idx]
        else:
            dataset = self

        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(dataset, batch_size, shuffle)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(dataset, batch_size, shuffle)
        else:
            raise NotImplementedError(f'Unsupported type {type}')

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--type', type=str, default='unlabeled')
    parser.add_argument('--data_name', type=str, default='ca-CondMat')
    parser.add_argument('--task', type=str, default='NC')
    parser.add_argument('--subgraph_type', type=str, default='ego') # ego | rw
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--out_channels', type=int, default=None)
    parser.add_argument('--feat_dim', type=int, default=64)
    parser.add_argument('--max_node_num', type=int, default=None)
    parser.add_argument('--sampling_mode', type=str, default=None) # None | random | shortest_path
    parser.add_argument('--force_process', action='store_true')
    parser.add_argument('--orig_feat', action='store_true')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--rank_nodes', type=str, default='feat_corr')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--fill_zeros', action='store_true')
    parser.add_argument('--thres', type=int, default=200)
    parser.add_argument('--num_threads', type=int, default=128)
    parser.add_argument('--return_list', action='store_true')
    args = parser.parse_args()
    print(args)
    torch.set_num_threads(args.num_threads)

    if args.type == 'downstream_induced': 
        stru_feat_principle = None if args.orig_feat else 'concat'
        dataset = DownstreamInducedDataset(task=args.task, data_name=args.data_name, subgraph_type=args.subgraph_type, 
                                           num_hops=args.num_hops, walk_length=args.walk_length, repeat=args.repeat,
                                           out_channels=args.out_channels, max_node_num=args.max_node_num,
                                           embedding_dim=args.feat_dim, force_process=args.force_process, 
                                           save=args.save, stru_feat_principle=stru_feat_principle, rank_nodes=args.rank_nodes)
    elif args.type == 'unlabeled':
        data_name_list = [args.data_name]
        # data_name_list = ['network_repository', 'planetoid']
        dataset = UnlabeledInducedDataset(data_name_list=data_name_list, subgraph_type=args.subgraph_type, 
                                          num_hops=args.num_hops, walk_length=args.walk_length, repeat=args.repeat, 
                                          embedding_dim=args.feat_dim, max_node_num=args.max_node_num, 
                                          force_process=args.force_process, save=args.save, 
                                          sampling_mode=args.sampling_mode)

    num_nodes = [g.num_nodes for g in dataset]
    print(f'num_graphs: {len(num_nodes)}')
    print(f'num_nodes: {np.mean(num_nodes)} +/- {np.std(num_nodes)}, max = {np.max(num_nodes)}, min = {np.min(num_nodes)}')
    # print(dataset.splits.keys())