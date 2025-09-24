import os
import math
import os.path as osp
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from glob import glob
from tqdm import tqdm
from scipy.io import mmread
from os.path import basename
from sklearn.model_selection import train_test_split, ShuffleSplit
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

import torch
import torch_geometric
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit, NormalizeFeatures
from torch_geometric.datasets import (
    Amazon, Flickr, Planetoid, WebKB, Actor, LINKXDataset, ZINC,
    WikipediaNetwork, DeezerEurope, HeterophilousGraphDataset
)
from torch_geometric.utils import (
    to_dense_adj,
    to_undirected,
    from_networkx,
    contains_isolated_nodes,
    remove_isolated_nodes,
    add_remaining_self_loops
)

from dataset.feature import get_features, aggregate_edge_attr
from dataset.subgraph import extract_subgraphs, extract_segments
from dataset.property import get_properties, split_by_property
from dataset.loader import (
    get_batched_datalist,
    MultiEpochsPYGDataLoader,
    graphs_to_feat_and_adj_dataloader
)
from dataset.misc import (
    str2int,
    preprocess,
    download_file,
    multiple_random_splits,
    read_link_prediction_data,
    get_diffusion_attributes,
)
from dataset.graph_prop_datasets import (
    NCI1, RedditBinary, Reddit5K, Reddit12K, Proteins, DD, Enzymes,
    IMDBBinary, IMDBMulti, Collab, GithubStargazers
)


FEATURE_CHOICE = ['node2vec', 'cn', 'aa', 'ra', 'ppr', 'katz']


# TODO: include downloading script
class NetworkRepositoryDataset(InMemoryDataset):
    META_PATH = '/egr/research-dselab/shared/dance/netrepo/netrepo.csv'
    DATA_ROOT = '/egr/research-dselab/shared/dance/netrepo/data'
    POSTFIXES = ['mtx', 'txt', 'txt.gz', 'edges']
    DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'ppr']
    EXTERNAL_DATASETS_DICT = {
        'github_stargazers': GithubStargazers,
        'Reddit2K': RedditBinary,
    }

    def __init__(
            self, dir_path='./data/misc/full_network_repository', deg_mean_max=3, density_max=0.1, entropy_max=10, 
            deg_var_max=3, num_edges_max=1e6, num_nodes_max=1e4, dense_num_nodes_max=0, nodes_and_edges=True, # filtering
            segment=False, thres=1000, fill_zeros=False, max_node_num=None, return_list=False, #seg
            feature_flag=False, feature_types=None, embedding_dim=60, walk_length_node2vec=20, 
            context_size=20, walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, 
            p_ppr=0.85, beta_katz=0.005, path_len=3, remove=False, aggr='add', aggr_repeat=10, #feat
            property_types=['all'], n_clusters=10, seed=10, scaler_dict=None, kmeans=None, # prop & kmeans
            external_datasets=None, extract_attributes=True, **kwargs
        ):

        self.dir_path = dir_path
        self.segment = segment
        self.thres = thres
        self.fill_zeros = fill_zeros
        self.max_node_num = max_node_num
        self.return_list = return_list
        self.external_datasets = external_datasets

        self.deg_mean_max = deg_mean_max
        self.density_max = density_max
        self.entropy_max = entropy_max
        self.deg_var_max = deg_var_max
        self.num_edges_max = num_edges_max
        self.num_nodes_max = num_nodes_max
        self.dense_num_nodes_max = dense_num_nodes_max
        self.nodes_and_edges = nodes_and_edges

        self.feature_flag = feature_flag
        if self.feature_flag and feature_types is None:
            feature_types = self.DEFAULT_FEATURE_TYPES
        self.feat_name = '_'.join(sorted(feature_types)) if feature_types is not None else 'none'
        self.feature_types = feature_types
        self.feat_config = {
            'embedding_dim': embedding_dim, 'walk_length': walk_length_node2vec, 'context_size': context_size,
            'walks_per_node': walks_per_node, 'p_node2vec': p_node2vec, 'q_node2vec': q_node2vec,
            'num_negative_samples': num_negative_samples, 'p_ppr': p_ppr, 'beta_katz': beta_katz,
            'path_len': path_len, 'remove': remove,
        }
        self.aggr = aggr
        self.aggr_repeat = aggr_repeat

        self.property_types = property_types
        self.scaler_dict = scaler_dict
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.seed = seed

        root = self.dir_path.replace('-', '_')
        super().__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict = torch.load(self.processed_paths[1])

        print('Extracting attributes for generation...')
        data_list = [self.get(i) for i in self.indices()]  #[:10]  ## temp
        for i in tqdm(range(len(data_list))):
            if extract_attributes:
                data_list[i] = get_diffusion_attributes(data_list[i])
            data_list[i].kmeans_labels = torch.from_numpy(self.kmeans_labels[i].repeat(data_list[i].num_nodes))
        self.data, self.slices = self.collate(data_list)
        
        del data_list

        self.max_degree = int(max(
            [torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self]
        ))

        self.max_degrees = []
        for i in range(self.kmeans.n_clusters):
            idx = np.where(self.kmeans_labels == i)[0] ## temp
            if len(idx) > 0:
                self.max_degrees.append(
                    int(max([torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self[idx]]))
                )
            else:
                self.max_degrees.append(1)

        self.splits = ['train'] * len(self)
    
    @property
    def processed_file_names(self):
        save_name = f"entr{self.entropy_max}_dens{self.density_max}_denm{int(self.dense_num_nodes_max)}"
        save_name += f"_degm{self.deg_mean_max}_degv{self.deg_mean_max}"
        save_name += f"_node{int(self.num_nodes_max)}_edge{int(self.num_edges_max)}"
        if self.feature_flag:
            save_name += f"-feat_{self.feat_name}"
        else:
            save_name += f"-feat_none"

        if self.segment:
            save_name += f'-thres_{self.thres}'
            if self.return_list:
                save_name += f'-fill_{str(self.fill_zeros)}'
                if self.fill_zeros:
                    save_name += f'-max_{self.max_node_num}'
        
        if self.external_datasets is not None:
            save_name += f"-ext_{'_'.join(sorted(self.external_datasets))}"

        split_name = save_name + f"-{'_'.join(sorted(self.property_types))}"
        split_name += f'_{self.n_clusters}clusters'

        if self.nodes_and_edges:
            split_name += '_ne'

        if self.segment and self.return_list:
            split_name += f'_seg-thres{self.thres}'
        
        save_name += '.pt'
        split_name += '.pt'
        file_names = [save_name, split_name]

        return file_names
    
    def read_data(self, root, data_name):
        files = os.listdir(root)
        data_file = [
            x for x in files 
            if any([x.endswith(pf) for pf in self.POSTFIXES]) 
            and not x.startswith('readme')
        ]
        if len(data_file) > 0:
            if len(data_file) > 1:
                print('######### data files more than 1 #########')
                print(f'######### data_name: {data_name} #########')
                print(data_file)

            data_file = data_file[0]
        
            flag = True
            try:
                edges_coo = mmread(f'{root}/{data_file}')
                edge_index = torch.stack(
                    (torch.from_numpy(edges_coo.col), torch.from_numpy(edges_coo.row))
                ).long()
                edge_attr = torch.from_numpy(edges_coo.data)
        
            except:
        
                try:
                    edges_df = pd.read_csv(f'{root}/{data_file}', header=None, sep=' ')
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
            
                    elif data_file.endswith('txt.gz'):
                        # remove irrelavent rows
                        counter = 0
                        while True:
                            try:
                                int(str(edges_df.iloc[counter, 0]).split('\t')[0])
                                break
                            except:
                                counter += 1
                        temp_df = edges_df.iloc[counter:, 0].copy()
                        edges_df = temp_df.str.split('\t', n=1, expand=True)
            
                    edge_index = torch.stack((
                        torch.tensor(edges_df.iloc[:,0].values.astype(int), dtype=torch.long), 
                        torch.tensor(edges_df.iloc[:,1].values.astype(int), dtype=torch.long)
                    ))
                    if edges_df.shape[1] > 2 and not any(edges_df.iloc[:,2].isna()):
                        edge_attr = torch.tensor(edges_df.iloc[:,2].values.astype(float))
                    else:
                        edge_attr = None
        
                except:
                    edge_index = None
                    edge_attr = None
                    flag = False

        else:
            print('######### data files empty #########')
            print(f'######### data_name: {data_name} #########')
            print(files)
            edge_index = None
            edge_attr = None
            flag = False

        if edge_index is not None and edge_index.numel() == 0:
            flag = False

        if flag and edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        return edge_index, edge_attr, flag

    def process(self):
        meta = pd.read_csv(self.META_PATH)
        meta = meta[~meta['Nodes'].isna()]
        meta = meta[~meta['Edges'].isna()]

        # str2int
        meta['Nodes'] = meta['Nodes'].apply(lambda x: str2int(x))
        meta['Edges'] = meta['Edges'].apply(lambda x: str2int(x))

        # for sanity
        meta = meta[~meta['Nodes'].isna()]
        meta = meta[~meta['Edges'].isna()]

        # remove large graphs
        submeta = meta[meta['Nodes'] <= 5e4]

        # set index
        submeta = submeta.drop_duplicates('Name')
        submeta = submeta.set_index('Name')

        # corrupted files
        # index_to_drop = ['scc-rt-islam', 'Maragal-6', 'Maragal-7', 'Maragal-8']
        index_to_drop = ['scc-rt-islam']
        submeta = submeta.drop(index_to_drop, axis=0)

        data_names = []
        data_list = []
        submeta['accessibility'] = True

        print('Reading data...')
        for name in tqdm(submeta.index):
            data_root = f'{self.DATA_ROOT}/{name}'
            edge_index, edge_attr, flag = self.read_data(data_root, name)
            submeta.loc[name, 'accessibility'] = flag
            if flag:
                if contains_isolated_nodes(edge_index):
                    edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr)

                if edge_index.numel() > 0:
                    data_names.append(name)
                    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
                    data = Data(
                        edge_index=edge_index, edge_attr=edge_attr, num_nodes=edge_index.max() + 1
                    ).coalesce()
                    data_list.append(data)
        
        print('Subseting graphs...')
        # remove empty graphs
        self.data_names = np.array(data_names)
        num_nodes = np.array([g.num_nodes for g in data_list])
        num_edges = np.array([g.num_edges for g in data_list])
        non_empty_mask = np.logical_and(np.array(num_nodes) != 0, np.array(num_edges) != 0)
        data_list = [data_list[j] for j in np.where(non_empty_mask)[0]]
        self.data_names = self.data_names[non_empty_mask]

        # subset according to properties
        num_nodes = np.array([g.num_nodes for g in data_list])
        num_edges = np.array([g.num_edges for g in data_list])
        data_list, property_attr, property_dict, scaler_dict = get_properties(data_list)
        masks = (
            (scaler_dict['deg_mean'].transform(property_dict['deg_mean'])) > self.deg_mean_max,
            np.logical_and(property_dict['density'] > self.density_max,  num_nodes > self.dense_num_nodes_max),
            (scaler_dict['entropy'].transform(property_dict['entropy'])) > self.entropy_max,
            (scaler_dict['deg_var'].transform(property_dict['deg_var'])) > self.deg_var_max,
            num_edges > self.num_edges_max,
            num_nodes > self.num_nodes_max,
        )
        disgard_mask = np.logical_or.reduce(masks)
        subset_mask = np.logical_not(disgard_mask)
        data_list = [data_list[j] for j in np.where(subset_mask)[0]]
        self.data_names = self.data_names[subset_mask]

        if self.external_datasets is not None:
            assert all([i in self.EXTERNAL_DATASETS_DICT for i in self.external_datasets])
            for data_name in self.external_datasets:
                dataset_class = self.EXTERNAL_DATASETS_DICT[data_name]
                dataset = dataset_class(outer_k=1, data_dir=f'data/graph_property_prediction')
                temp_data_list = dataset.dataset.data
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(range(len(temp_data_list)))
                for i in perm[:min(len(temp_data_list), 1000)]:
                    temp_data = temp_data_list[i]
                    edge_index, edge_attr = temp_data.edge_index, temp_data.edge_attr
                    if edge_attr is None:
                        edge_attr = torch.ones(edge_index.shape[1])

                    if contains_isolated_nodes(edge_index):
                        edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr)

                    if edge_index.numel() > 0:
                        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
                    
                    data = Data(
                        edge_index=edge_index, edge_attr=edge_attr, num_nodes=edge_index.max() + 1
                    ).coalesce()
                    data_list.append(data)

        # calculate structure properties
        data_list, property_attr, self.property_dict, self.scaler_dict = get_properties(
            data_list, self.property_types, self.scaler_dict, nodes_and_edges=self.nodes_and_edges
        )
        self.kmeans, self.kmeans_labels = split_by_property(property_attr, self.kmeans, self.n_clusters, self.seed)

        self.data, self.slices = self.collate(data_list)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices), self.processed_paths[0])
        torch.save((self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict), 
                   self.processed_paths[1])
    
    def get_dataloader(self, type='pyg', nodes_max=None, edges_max=None, batch_size=1, shuffle=False,
                       loader_kwargs={}, return_dataset=False, **kwargs):
        dataset = get_batched_datalist(self, nodes_max=nodes_max, edges_max=edges_max)
        if nodes_max is not None or edges_max is not None:
            for temp_data in dataset:
                temp_data.nodes_per_graph = temp_data.nodes_per_graph.sum()
                temp_data.edges_per_graph = temp_data.edges_per_graph.sum()

        if return_dataset:
            return dataset

        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(dataset, batch_size, shuffle, **loader_kwargs)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(dataset, batch_size, shuffle, **loader_kwargs)
        else:
            raise NotImplementedError(f'Unsupported type {type}')


class GenericDataset(InMemoryDataset):
    DATASETS = ['ego', 'community']
    RAW_FILE_PATH = 'data/misc/raw'
    DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'aa', 'ppr', 'katz']

    def __init__(self, data_name='ego', dir_path='./data', degree_flag=True, feature_flag=False, embedding_dim=60, 
                 walk_length_node2vec=20, context_size=20, walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, 
                 num_negative_samples=1, p_ppr=0.85, beta_katz=0.005, path_len=3, remove=False, seed=10,
                 aggr='add', aggr_repeat=10, **kwargs):
        self.data_name = data_name
        self.dir_path = dir_path
        self.degree_flag = degree_flag
        self.seed = seed

        self.feature_types = self.DEFAULT_FEATURE_TYPES if feature_flag else None
        self.feat_name = '_'.join(sorted(self.feature_types)) if self.feature_types is not None else 'none'
        self.feat_params = {
            'embedding_dim': embedding_dim, 'walk_length': walk_length_node2vec, 'context_size': context_size,
            'walks_per_node': walks_per_node, 'p_node2vec': p_node2vec, 'q_node2vec': q_node2vec,
            'num_negative_samples': num_negative_samples, 'p_ppr': p_ppr, 'beta_katz': beta_katz,
            'path_len': path_len, 'remove': remove,
        }
        self.aggr = aggr
        self.aggr_repeat = aggr_repeat

        root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
        super().__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')
        self.data, self.slices, self.splits = torch.load(self.processed_paths[0])
        self.max_degree = int(max(
            [torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self]
        ))

    @property
    def processed_file_names(self):
        save_name = f'{self.data_name}-feat_{self.feat_name}.pt'
        return [save_name]

    def process(self):
        nx_graphs = pickle.load(open(f"{self.RAW_FILE_PATH}/{self.data_name}.pkl", 'rb'))

        data_list = []
        for nx_graph in nx_graphs:
            data = preprocess(nx_graph, degree=self.degree_flag)
            data_list.append(data)

        # structure features
        if self.feature_types is not None:
            for data in tqdm(data_list):
                feat_dict = get_features(
                    data.edge_index, data.edge_weight, data.num_nodes, self.feature_types,
                    print_loss=False, return_dict=True, **self.feat_params
                )
                node_feat = feat_dict.pop('node2vec').nan_to_num()
                edge_attr = torch.cat(list(feat_dict.values()), dim=1).nan_to_num() if len(feat_dict) > 0 else None
                if edge_attr is not None:
                    structure_feat = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                         repeat=self.aggr_repeat, aggr=self.aggr)
                else:
                    structure_feat = node_feat

                # sanity check for structure feature
                data.x = torch.nan_to_num(structure_feat)

        self.data, self.slices = self.collate(data_list)
        self.get_splits(self.seed)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices, self.splits), self.processed_paths[0])

    def get_splits(self, seed=10):
        l = len(self)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(np.arange(l))
        self.splits = {
            'train': perm[:int(0.8*l)],
            'test': perm[int(0.8*l):]
        }

    def get_split_dataset(self, split):
        if not hasattr(self, 'splits'):
            self.get_splits()
        assert split in self.splits
        return self[self.splits[split]]

    def get_dataloader(self, type='pyg', split=None, batch_size=64, shuffle=False, loader_kwargs={}, **kwargs):
        if split is not None:
            dataset = self.get_split_dataset(split)
        else:
            dataset = self

        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(dataset, batch_size, shuffle, **loader_kwargs)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(dataset, batch_size, shuffle, **loader_kwargs)
        else:
            raise NotImplementedError(f'Unsupported type {type}')


class LinkPredictionDataset(InMemoryDataset):
    # some datasets are collected from https://noesis.ikor.org/datasets/link-prediction
    DATASETS = ['cora', 'citeseer', 'pubmed', 'ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2',
                'power', 'yst', 'erd', 'photo', 'flickr']
    DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'ppr']

    def __init__(self, data_name='cora', dir_path='./data', feature_flag=False, feature_types=None, embedding_dim=60, 
                 walk_length_node2vec=20, context_size=20, walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, 
                 num_negative_samples=1, p_ppr=0.85, beta_katz=0.005, path_len=3, remove=False,
                 aggr='add', aggr_repeat=10, segment=False, thres=1000, fill_zeros=False,
                 max_node_num=None, return_list=False, num_hops=0, property_types=['all'], nodes_and_edges=True,
                 n_clusters=8, seed=10, scaler_dict=None, kmeans=None, **kwargs):
        self.data_name = data_name
        self.dir_path = f'{dir_path}/link_prediction'
        self.segment = segment
        self.thres = thres
        self.fill_zeros = fill_zeros
        self.max_node_num = max_node_num
        self.return_list = return_list
        self.num_hops = num_hops

        self.feature_flag = feature_flag
        if self.feature_flag and feature_types is None:
            feature_types = self.DEFAULT_FEATURE_TYPES
        self.feature_types = feature_types
        self.feat_name = '_'.join(sorted(self.feature_types)) if self.feature_flag else 'none'
        self.feat_params = {
            'embedding_dim': embedding_dim, 'walk_length': walk_length_node2vec, 'context_size': context_size,
            'walks_per_node': walks_per_node, 'p_node2vec': p_node2vec, 'q_node2vec': q_node2vec,
            'num_negative_samples': num_negative_samples, 'p_ppr': p_ppr, 'beta_katz': beta_katz,
            'path_len': path_len, 'remove': remove,
        }
        self.aggr = aggr
        self.aggr_repeat = aggr_repeat

        self.property_types = property_types
        self.scaler_dict = scaler_dict
        self.nodes_and_edges = nodes_and_edges
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.seed = seed

        root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
        super().__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict = torch.load(self.processed_paths[1])

        self.max_degree = int(max(
            [torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self]
        ))

        self.max_degrees = []
        for i in range(self.kmeans.n_clusters):
            idx = np.where(self.kmeans_labels == i)[0]
            if len(idx) > 0:
                self.max_degrees.append(
                    int(max([torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self[idx]]))
                )
            else:
                self.max_degrees.append(1)

    @property
    def processed_file_names(self):
        save_name = f'{self.data_name}-feat_{self.feat_name}'
        if self.segment:
            save_name += f'-thres_{self.thres}'
            if self.num_hops > 0:
                save_name += f'-{self.num_hops}_hop'
            if self.max_node_num is not None:
                save_name += f'-max_{self.max_node_num}'
            if self.return_list:
                save_name += f'-fill_{str(self.fill_zeros)}'

        split_name = save_name + f"-{'_'.join(sorted(self.property_types))}"
        if self.nodes_and_edges:
            split_name += '_ne'

        if self.scaler_dict is not None:
            split_name += '_pre_scale'

        if self.kmeans is not None:
            split_name += '_pre_kmeans'
        else:
            split_name += f'_{self.n_clusters}clusters'
        
        if self.segment and self.return_list:
            split_name += f'_seg-thres{self.thres}'

        save_name += '.pt'
        split_name += '.pt'
        file_names = [save_name, split_name]

        return file_names

    def process(self):
        if self.data_name in ['cora', 'citeseer', 'pubmed']:
            data = read_link_prediction_data(self.data_name, self.dir_path)

        elif self.data_name.startswith('ogbl'):
            dataset = PygLinkPropPredDataset(name=self.data_name, root=f'{self.dir_path}/{self.data_name}'.replace('-', '_'))
            data = dataset[0]
            split_edge = dataset.get_edge_split()

            if self.data_name == 'ogbl-citation2':
                source, target = split_edge['train']['source_node'], split_edge['train']['target_node']
                data['train_pos'] = torch.stack([source, target])

                source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
                data['valid_pos'] = torch.stack([source, target])
                data['valid_neg'] = split_edge['valid']['target_node_neg']

                source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
                data['test_pos'] = torch.stack([source, target])
                data['test_neg'] = split_edge['test']['target_node_neg']
            
            else:
                data['train_pos'] = split_edge['train']['edge'].T
                data['valid_pos'] = split_edge['valid']['edge'].T
                data['valid_neg'] = split_edge['valid']['edge_neg'].T
                data['test_pos'] = split_edge['test']['edge'].T
                data['test_neg'] = split_edge['test']['edge_neg'].T

            idx = torch.randperm(data['train_pos'].size(1))[:data['valid_pos'].size(1)]
            data['train_val'] = data['train_pos'][:, idx]
        
        else:
            if self.data_name == 'power':
                raw_file = './data/link_prediction/noesis/UPG_full.net'
                data_nx = nx.read_pajek(raw_file)
                data = from_networkx(data_nx)
                data.edge_index = to_undirected(data.edge_index)
            
            elif self.data_name == 'yst':
                raw_file = './data/link_prediction/noesis/YST_full.net'
                data_nx = nx.read_pajek(raw_file)
                data = from_networkx(data_nx)
                data.edge_index = to_undirected(data.edge_index)

            elif self.data_name == 'erd':
                raw_file = './data/link_prediction/noesis/ERD_full.net'
                data_nx = nx.read_pajek(raw_file)
                data = from_networkx(data_nx)
                data.edge_index = to_undirected(data.edge_index)

            elif self.data_name == 'photo':
                data = Amazon(self.dir_path, 'Photo')[0]
            
            elif self.data_name == 'flickr':
                data = Flickr(f'{self.dir_path}/{self.data_name}')[0]
            
            else:
                raise NotImplementedError(f'Unsupported dataset {self.data_name}')

            transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.1,
                                        add_negative_train_samples=True)
            
            train_data, val_data, test_data = transform(data)
            data.edge_index, data.edge_weight = train_data.edge_index, train_data.edge_weight

            data['train_pos'] = train_data.edge_label_index[:, train_data.edge_label == 1]

            data['valid_pos'] = val_data.edge_label_index[:, val_data.edge_label == 1]
            data['valid_neg'] = val_data.edge_label_index[:, val_data.edge_label == 0]

            data['test_pos'] = test_data.edge_label_index[:, test_data.edge_label == 1]
            data['test_neg'] = test_data.edge_label_index[:, test_data.edge_label == 0]

            idx = torch.randperm(data['train_pos'].size(1))[:data['valid_pos'].size(1)]
            data['train_val'] = data['train_pos'][:, idx]

        if data.x is not None:
            data.x = data.x.to(torch.float32)

        data.adj = SparseTensor.from_edge_index(
            data.edge_index, data.edge_weight, [data.num_nodes, data.num_nodes]
        )

        # attributes for sampling
        # adj = to_dense_adj(data.edge_index)[0].long()
        # row, col = torch.triu_indices(data.num_nodes, data.num_nodes,1)
        # data.full_edge_index = torch.stack([row, col])
        # data.full_edge_attr = adj[data.full_edge_index[0], data.full_edge_index[1]]
        # data.nodes_per_graph = data.num_nodes
        # data.edges_per_graph = data.num_nodes * (data.num_nodes - 1) // 2

        # structure features
        if self.feature_flag:
            feat_dict = get_features(
                data.edge_index, data.edge_weight, data.num_nodes, self.feature_types,
                print_loss=False, return_dict=True, **self.feat_params
            )
            node_feat = feat_dict.pop('node2vec').nan_to_num()
            edge_attr = torch.cat(list(feat_dict.values()), dim=1).nan_to_num() if len(feat_dict) > 0 else None
            if edge_attr is not None:
                structure_feat = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                     repeat=self.aggr_repeat, aggr=self.aggr)
            else:
                structure_feat = node_feat

            # sanity check for structure feature
            data.stru_feat = torch.nan_to_num(structure_feat)
        
        if self.segment:
            out = extract_segments(data, thres=self.thres, fill_zeros=self.fill_zeros, return_list=self.return_list,
                                   max_node_num=self.max_node_num, num_hops=self.num_hops)
            data_list = out if self.return_list else [out]
        else:
            data_list = [data]

        data_list, property_attr, self.property_dict, self.scaler_dict = get_properties(
            data_list, self.property_types, self.scaler_dict, nodes_and_edges=self.nodes_and_edges
        )
        self.kmeans, self.kmeans_labels = split_by_property(property_attr, self.kmeans, self.n_clusters, self.seed)
        
        self.data, self.slices = self.collate(data_list)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices), self.processed_paths[0])
        torch.save((self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict), 
                   self.processed_paths[1])

    def get_dataloader(self, type='pyg', split=None, batch_size=64, shuffle=False, **kwargs):
        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(self, batch_size, shuffle)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(self, batch_size, shuffle)
        else:
            raise NotImplementedError(f'Unsupported type {type}')


class GraphPropertyPredictionDataset(InMemoryDataset):
    # some datasets are collected from https://github.com/GRAND-Lab/graph_datasets
    # some datasets are collected from https://github.com/diningphil/gnn-comparison
    # Reddit_multi_12k is collected from https://github.com/yunshengb/UGraphEmb
    DATASETS = [
        'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2', 'Brain', 'DBLP_v1', 'NCI_balanced',
        'NCI_full', 'PTC_mtl', 'PTC_pn', 'Twitter_Graph', 'NCI1', 'DD', 'Enzymes', 'Proteins',
        'Reddit_binary', 'Reddit_multi_5k', 'IMDB_binary', 'IMDB_multi', 'Collab', 'Reddit_multi_12k'
    ]
    GRAPH_PROP_DATASETS_DICT = {
        'Reddit_binary': RedditBinary,
        'Reddit_multi_5k': Reddit5K,
        'Reddit_multi_12k': Reddit12K,
        'Collab': Collab,
        'IMDB_binary': IMDBBinary,
        'IMDB_multi': IMDBMulti,
        'NCI1': NCI1,
        'Enzymes': Enzymes,
        'Proteins': Proteins,
        'DD': DD,
        'github_stargazers': GithubStargazers,
    }
    RAW_PATH_DICT = {
        'Reddit_multi_12k': 'data/graph_property_prediction/RedditMulti12k/'
    }

    def __init__(self, data_name='Enzymes', dir_path='./data', property_types=['all'], nodes_and_edges=True, 
                 n_clusters=8, seed=10, scaler_dict=None, kmeans=None, add_self_loop=True, extract_attributes=False, 
                 one_hot=True, **kwargs):
        self.data_name = data_name
        self.dir_path = f'{dir_path}/graph_property_prediction'

        self.property_types = property_types
        self.scaler_dict = scaler_dict
        self.nodes_and_edges = nodes_and_edges
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.seed = seed

        root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
        super().__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')
        self.data, self.slices, self.splits = torch.load(self.processed_paths[0])
        self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict = torch.load(self.processed_paths[1])

        self.n_classes = len(self.y.unique())
        data_list = [self.get(i) for i in self.indices()]  #[:10]  ## temp
        for i in tqdm(range(len(data_list))):
            data_list[i].kmeans_labels = torch.from_numpy(self.kmeans_labels[i].repeat(data_list[i].num_nodes))
            if one_hot:
                data_list[i].labels = torch.nn.functional.one_hot(data_list[i].y.long(), num_classes=self.n_classes).float()
            if add_self_loop:
                data_list[i].edge_index, _ = add_remaining_self_loops(data_list[i].edge_index)
            if extract_attributes:
                data_list[i] = get_diffusion_attributes(data_list[i])

        self.data, self.slices = self.collate(data_list)
        del data_list

        self.max_degree = int(max(
            [torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self]
        ))

        self.max_degrees = []
        for i in range(self.kmeans.n_clusters):
            idx = np.where(self.kmeans_labels == i)[0] ## temp
            if len(idx) > 0:
                self.max_degrees.append(
                    int(max([torch_geometric.utils.degree(g.edge_index[0]).max().item() for g in self[idx]]))
                )
            else:
                self.max_degrees.append(1)

    @property
    def processed_file_names(self):
        save_name = f'{self.data_name}'

        split_name = save_name + f"-{'_'.join(sorted(self.property_types))}"
        if self.nodes_and_edges:
            split_name += '_ne'

        if self.scaler_dict is not None:
            split_name += '_pre_scale'

        if self.kmeans is not None:
            split_name += '_pre_kmeans'
        else:
            split_name += f'_{self.n_clusters}clusters'

        save_name += '.pt'
        split_name += '.pt'
        file_names = [save_name, split_name]

        return file_names

    def read_data(filepath):
        df = pd.read_csv(filepath, header=None)
        df[['type', 'val']] = df.iloc[:,0].str.split(' ', n=1, expand=True)
        ptrs = np.where(df['type'] == 'x')[0]

        data_list = []
        for i in range(len(ptrs)):
            if i == 0:
                sub_df = df[:(ptrs[i] + 1)]
            else:
                sub_df = df[(ptrs[i-1] + 1):(ptrs[i] + 1)]

            nodes = sub_df[sub_df['type'] == 'n']
            x = nodes['val'].str.split(' ', n=1, expand=True).values[:,1:]
            # x[0] = x[0].astype(int)
            # x = x.set_index(0)
            # x = x.loc[range(len(x))]

            try:
                x = torch.from_numpy(x.astype(float))
            except:
                # x = None
                pass

            edges = sub_df[sub_df['type'] == 'e']
            edges[['src', 'dst', 'label']] = edges['val'].str.split(' ', n=2, expand=True)
            edge_index = np.stack((edges['src'].values.astype(int), edges['dst'].values.astype(int)))
            edge_index = torch.from_numpy(edge_index)
            edge_label = edges['label'].values

            y = torch.from_numpy(sub_df[sub_df['type'] == 'x']['val'].values.astype(float))

            edge_index = to_undirected(edge_index)
            data_list.append(Data(x=x, edge_index=edge_index, edge_label=edge_label, y=y))

        return data_list
    
    def iterate_get_graphs(self, dir):
        graphs = []
        for file in glob(dir + '/*.gexf'):
            gid = int(basename(file).split('.')[0])
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            graphs.append(from_networkx(g))
            if not nx.is_connected(g):
                print('{} not connected'.format(gid))
        
        graph_idx = [data.gid.item() for data in graphs]
        return graphs, graph_idx

    def process(self):
        if self.data_name in self.GRAPH_PROP_DATASETS_DICT.keys():
            dataset_class = self.GRAPH_PROP_DATASETS_DICT[self.data_name]

            kwargs = {'data_dir': self.dir_path}
            if self.data_name == 'Enzymes':
                kwargs['use_node_attrs'] = True

            if self.data_name in ['Reddit_binary', 'Reddit_multi_5k', 'Reddit_multi_12k', 'Collab',
                                  'github_stargazers', 'IMDB_binary', 'IMDB_multi']:
                kwargs['use_node_degree'] = True

            dataset = dataset_class(**kwargs)
            data_list, splits = dataset.dataset.data, dataset.splits
            self.splits = [
                {
                    'test': s['test'], 'train': s['model_selection'][0]['train'], 
                    'valid': s['model_selection'][0]['validation']
                } for s in splits
            ]

        elif self.data_name.startswith('ogbg'):
            dataset = PygGraphPropPredDataset(name=self.data_name, root=f'{self.dir_path}/{self.data_name}'.replace('-', '_'))
            data_list, self.splits = [dataset.get(i) for i in dataset.indices()], dataset.get_idx_split()

        elif self.data_name == "zinc":
            dataset = ZINC(root=f'{self.dir_path}/{self.data_name}'.replace('-', '_'), subset=True)
            data_list = [dataset.get(i) for i in dataset.indices()]
            self.splits = np.array(['train'] * len(dataset))

        else:
            data_list = self.read_data(self.data_name)

        data_list, property_attr, self.property_dict, self.scaler_dict = get_properties(
            data_list, self.property_types, self.scaler_dict, nodes_and_edges=self.nodes_and_edges
        )
        self.kmeans, self.kmeans_labels = split_by_property(property_attr, self.kmeans, self.n_clusters, self.seed)

        self.data, self.slices = self.collate(data_list)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices, self.splits), self.processed_paths[0])
        torch.save((self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict), 
                   self.processed_paths[1])


class NodeInducedDataset(InMemoryDataset):

    def __init__(self, data_name='cora', dir_path='./data', load_fix_split=True, full_to_undirected=False,
                 subgraph_type='ego', num_hops=2, walk_length=10, repeat=5, max_node_num=None,
                 fill_zeros=False, add_self_loop=False, rank_nodes='feat_corr', sub_to_undirected=False, # subgraph params
                 property_types=['all'], nodes_and_edges=True, n_clusters=8, seed=10, 
                 scaler_dict=None, kmeans=None, # self-labeling params
                 extract_attributes=False, one_hot=True, filter_nodes=True, keep_labels=False, **kwargs):

        self.data_name = data_name
        self.dir_path = f'{dir_path}/node_classification'
        self.load_fix_split = load_fix_split
        self.full_to_undirected = full_to_undirected
        self.sub_to_undirected = sub_to_undirected
        self.filter_nodes = filter_nodes
        self.keep_labels = keep_labels

        self.property_types = property_types
        self.scaler_dict = scaler_dict
        self.nodes_and_edges = nodes_and_edges
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.seed = seed

        self.subgraph_type = subgraph_type
        self.max_node_num = max_node_num
        self.subgraph_params = {
            'subgraph_type': subgraph_type, 'num_hops': num_hops, 'walk_length': walk_length,
            'repeat': repeat, 'max_node_num': max_node_num, 'fill_zeros': fill_zeros, 
            'add_self_loop': add_self_loop, 'rank_nodes': rank_nodes, 'make_undirected': sub_to_undirected,
        }

        root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
        super().__init__(root)
        print(f'Loading data from {self.processed_paths[0]}')
        self.data, self.slices, self.splits = torch.load(self.processed_paths[0])
        self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict = torch.load(self.processed_paths[1])

        self.n_classes = len(self.y.unique())
        data_list = [self.get(i) for i in self.indices()]  #[:10]  ## temp
        for i in tqdm(range(len(data_list))):
            data_list[i].kmeans_labels = torch.from_numpy(self.kmeans_labels[i].repeat(data_list[i].num_nodes))
            if one_hot:
                data_list[i].labels = torch.nn.functional.one_hot(data_list[i].y, num_classes=self.n_classes).float()
            if add_self_loop:
                data_list[i].edge_index, _ = add_remaining_self_loops(data_list[i].edge_index)
            if extract_attributes:
                data_list[i] = get_diffusion_attributes(data_list[i])

        self.data, self.slices = self.collate(data_list)
        del data_list

    @property
    def processed_file_names(self):
        save_name = f'{self.data_name}'
        if self.data_name in ['chameleon', 'squirrel'] and self.filter_nodes:
            save_name += '_filtered'

        if self.keep_labels:
            save_name += '_labels'

        if not self.load_fix_split:
            save_name += '_random'

        if self.full_to_undirected:
            save_name += '_full_to_undirected'
        elif self.sub_to_undirected:
            save_name += '_sub_to_undirected'

        if self.max_node_num is not None:
            max_node_num = self.subgraph_params['max_node_num']
            rank_nodes = self.subgraph_params['rank_nodes']
            save_name += f'-max_{max_node_num}-subsample_{rank_nodes}'

        if self.subgraph_type == 'ego':
            save_name += f"-{self.subgraph_params['num_hops']}_hop"
        elif self.subgraph_type == 'rw':
            save_name += f"-length_{self.subgraph_params['walk_length']}-repeat_{self.subgraph_params['repeat']}"

        split_name = save_name + f"-{'_'.join(sorted(self.property_types))}"
        if self.nodes_and_edges:
            split_name += '_ne'

        if self.scaler_dict is not None:
            split_name += '_pre_scale'

        if self.kmeans is not None:
            split_name += '_pre_kmeans'
        else:
            split_name += f'_{self.n_clusters}clusters'

        save_name += '.pt'
        split_name += '.pt'
        file_names = [save_name, split_name]

        return file_names
    
    def process(self):
        if self.data_name in ['citeseer', 'pubmed', 'cora']:
            dataset = Planetoid(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())
        
        elif self.data_name in ['cornell', 'texas', 'wisconsin']: # 10 splits 
            dataset = WebKB(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())

        elif self.data_name in ['chameleon', 'squirrel']: # 10 splits 
            if self.filter_nodes:
                url = f'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{self.data_name}_filtered.npz'
                root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
                fname = f'{root}/raw/{self.data_name}_filtered_directed.npz'
                download_file(url, fname)
                raw_data = np.load(fname, allow_pickle=True)

                data_dict = {}
                data_dict['x'] = torch.tensor(raw_data['node_features'], dtype=torch.float32)
                data_dict['y'] = torch.tensor(raw_data['node_labels'])
                data_dict['edge_index'] = torch.tensor(raw_data['edges']).t()

                data_dict['train_mask'] = torch.tensor(raw_data['train_masks']).t()
                data_dict['val_mask'] = torch.tensor(raw_data['val_masks']).t()
                data_dict['test_mask'] = torch.tensor(raw_data['test_masks']).t()

                data = Data(**data_dict)
                data = NormalizeFeatures()(data)

                dataset = [data]

            else:
                dataset = WikipediaNetwork(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())

        elif self.data_name in ['actor']: # 10 splits 
            root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
            dataset = Actor(root=root, transform=NormalizeFeatures())

        elif self.data_name in ['deezer_europe']: # 5 splits 
            root = f'{self.dir_path}/{self.data_name}'.replace('-', '_')
            dataset = DeezerEurope(root=root, transform=NormalizeFeatures())
            url = 'https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/splits/deezer-europe-splits.npy'
            download_file(url, f'{root}/raw/deezer-europe-splits.npy')
            self.splits = np.load(f'{root}/raw/deezer-europe-splits.npy', allow_pickle=True)

        elif self.data_name in ['penn94', 'genius']:  # 5 splits
            dataset = LINKXDataset(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())
        
        elif self.data_name in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
            dataset = HeterophilousGraphDataset(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())

        elif self.data_name.startswith('ogbn'):
            dataset = PygNodePropPredDataset(name=self.data_name, root=self.dir_path, transform=NormalizeFeatures())
            self.splits = [dataset.get_idx_split()]

        else:
            raise NotImplementedError(f'Unknown dataset {self.data_name}')

        try:
            data = dataset.get(0)
        except:
            data = dataset[0]

        if self.full_to_undirected:
            data.edge_index = to_undirected(data.edge_index)

        if not hasattr(self, 'splits') or not self.load_fix_split:
            if not self.load_fix_split or not hasattr(data, 'train_mask'):
                data.train_mask, data.val_mask, data.test_mask = multiple_random_splits(data.y)

            if len(data.train_mask.shape) == 1:
                keys = ['train_mask', 'val_mask', 'test_mask']
                for k in keys:
                    data[k] = data[k].unsqueeze(-1)

            self.splits = []
            for i in range(data.train_mask.shape[1]):
                self.splits.append({
                    'train': torch.where(data.train_mask[:, i])[0],
                    'valid': torch.where(data.val_mask[:, i])[0],
                    'test': torch.where(data.test_mask[:, i])[0],
                })

        nodes_list = [[x] for x in range(data.num_nodes)]
        y = data.y.long()
        subgraph_params = self.subgraph_params

        node_feat_dict = {}
        if self.keep_labels:
            node_feat_dict['node_labels'] = y.unsqueeze(-1)
        subgraph_params['node_feat_dict'] = node_feat_dict
        data_list = extract_subgraphs(data, y, nodes_list, **subgraph_params)

        data_list, property_attr, self.property_dict, self.scaler_dict = get_properties(
            data_list, self.property_types, self.scaler_dict, nodes_and_edges=self.nodes_and_edges
        )
        self.kmeans, self.kmeans_labels = split_by_property(property_attr, self.kmeans, self.n_clusters, self.seed)

        self.data, self.slices = self.collate(data_list)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices, self.splits), self.processed_paths[0])
        torch.save((self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict), 
                   self.processed_paths[1])


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--type', type=str, default='generic')
    parser.add_argument('--data_name', type=str, default='ego')
    parser.add_argument('--dir_path', type=str, default='./data/misc')
    parser.add_argument('--feature_flag', action='store_true')
    parser.add_argument('--num_threads', type=int, default=128)
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--thres', type=int, default=1000)
    parser.add_argument('--return_list', action='store_true')
    parser.add_argument('--num_hops', type=int, default=0)
    parser.add_argument('--to_segments', action='store_true')
    parser.add_argument('--get_prop', action='store_true')
    parser.add_argument('--num_edges_max', type=int, default=50000)
    parser.add_argument('--num_nodes_max', type=int, default=10000)
    parser.add_argument('--dense_num_nodes_max', type=int, default=100)
    parser.add_argument('--external_datasets', type=str, default=None, nargs='+')
    parser.add_argument('--subgraph_type', type=str, default='ego')
    parser.add_argument('--load_fix_split', action='store_true')
    parser.add_argument('--extract_attributes', action='store_true')
    parser.add_argument('--full_to_undirected', action='store_true')
    parser.add_argument('--sub_to_undirected', action='store_true')
    parser.add_argument('--filter_nodes', action='store_true')
    parser.add_argument('--keep_labels', action='store_true')
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    extra_configs = {}
    if args.get_prop:
        from dataset.property import get_properties, k_means

        kmeans, _, _, scaler_dict = torch.load(
            # 'data/misc/large_network_repository_network_repository_snap/processed/all_8clusters_seg-thres1000.pt'
            # 'data/misc/full_network_repository/processed/entr10_dens0.1_denm0_degm3_degv3_node4000_edge50000-feat_none-all_10clusters_ne.pt'
            'data/misc/full_network_repository/processed/entr10_dens0.1_denm50_degm3_degv3_node4000_edge50000-feat_none-ext_github_stargazers-all_10clusters_ne.pt'
        )
        extra_configs['kmeans'] = kmeans
        extra_configs['scaler_dict'] = scaler_dict

    if args.type == 'generic':
        dataset = GenericDataset(data_name=args.data_name, dir_path=args.dir_path, feature_flag=args.feature_flag)
    elif args.type == 'link_pred':
        dataset = LinkPredictionDataset(
            data_name=args.data_name, feature_flag=args.feature_flag, segment=args.segment, thres=args.thres,
            return_list=args.return_list, num_hops=args.num_hops, **extra_configs
        )
    elif args.type == 'nr':
        dataset = NetworkRepositoryDataset(
            num_edges_max=args.num_edges_max, num_nodes_max=args.num_nodes_max, 
            dense_num_nodes_max=args.dense_num_nodes_max, segment=args.segment, 
            thres=args.thres, return_list=args.return_list, num_hops=args.num_hops,
            external_datasets=args.external_datasets
        )
    elif args.type == 'graph_prop':
        dataset = GraphPropertyPredictionDataset(
            data_name=args.data_name, extract_attributes=args.extract_attributes, **extra_configs
        )
    elif args.type == 'node_subgraph':
        dataset = NodeInducedDataset(
            data_name=args.data_name, subgraph_type=args.subgraph_type, full_to_undirected=args.full_to_undirected,
            extract_attributes=args.extract_attributes, sub_to_undirected=args.sub_to_undirected, 
            filter_nodes=args.filter_nodes, keep_labels=args.keep_labels, **extra_configs
        )

    num_nodes = np.array([g.num_nodes for g in dataset])
    num_edges = np.array([g.num_edges for g in dataset])
    degrees = torch.cat([torch_geometric.utils.degree(g.edge_index[0]) for g in dataset])
    print(f'num_graphs: {len(dataset)}')
    print(f'num_nodes: {np.mean(num_nodes)} +/- {np.std(num_nodes)}, max = {np.max(num_nodes)}, min = {np.min(num_nodes)}')
    print(f'num_edges: {np.mean(num_edges)} +/- {np.std(num_edges)}, max = {np.max(num_edges)}, min = {np.min(num_edges)}')
    print(f'node degrees: {torch.mean(degrees)} +/- {torch.std(degrees)}, max = {torch.max(degrees)}, min = {torch.min(degrees)}')

    if args.to_segments:
        from dataset.subgraph import graph_to_segments

        data = dataset[0]
        data_list, remaining_edge_index, remaining_edge_attr = graph_to_segments(data, add_diff_attr=True)

        print(f'num_graphs: {len(data_list)}')
        num_nodes = [g.num_nodes for g in data_list]
        print(f'num_nodes: {np.mean(num_nodes)} +/- {np.std(num_nodes)}, max = {np.max(num_nodes)}, min = {np.min(num_nodes)}')

        num_edges = [g.num_edges for g in data_list]
        print(f'num_edges: {np.mean(num_edges)} +/- {np.std(num_edges)}, max = {np.max(num_edges)}, min = {np.min(num_edges)}')


    import ipdb; ipdb.set_trace()