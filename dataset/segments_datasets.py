import warnings
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
    to_dense_adj,
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
from dataset.property import get_properties, split_by_property
from dataset.loader import (
    MultiEpochsPYGDataLoader,
    graphs_to_feat_and_adj_dataloader
)
from dataset.misc import (
    download_file,
    download_unzip,
    read_link_prediction_data,
)


FEATURE_CHOICE = ['node2vec', 'cn', 'aa', 'ra', 'ppr', 'katz']
# DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'aa', 'ppr', 'katz']
DEFAULT_FEATURE_TYPES = ['node2vec', 'cn', 'ppr']


class SegmentsDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
    
    @property
    def processed_file_names(self):
        pass

    def _split(self, data_list, split_dict={'train': 0.9, 'valid': 0.1}, seed=10):
        assert sum([split_dict[k] for k in split_dict.keys()]) == 1
        N = len(data_list)
        splits = np.array(['train'] * N)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(range(N))
        splits[
            perm[
                int(N * split_dict['train']):int(N * (split_dict['train'] + split_dict['valid']))
            ]
        ] = 'valid'
        if 'test' in split_dict:
            splits[perm[int(N * (split_dict['train'] + split_dict['valid'])):]] = 'test'
        
        return splits

    def _get_full_edge(self, data_list):
        for d in data_list:
            adj = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes)[0].long()
            row, col = torch.triu_indices(d.num_nodes, d.num_nodes, 1)
            d.full_edge_index = torch.stack([row, col])
            d.full_edge_attr = adj[d.full_edge_index[0], d.full_edge_index[1]]
            d.nodes_per_graph = d.num_nodes
            d.edges_per_graph = d.num_nodes * (d.num_nodes - 1) // 2

        return data_list

    def get_split_dataset(self, split):
        split = [split] if not isinstance(split, list) else split
        split_array = np.array(self.splits)
        idx = []
        for s in split:
            assert s in split_array, f'{s} not in {split_array}'
            idx.extend(np.where(split_array == s)[0])
        return self[idx]
    
    def get_cluster_dataset(self, cluster):
        cluster = [cluster] if not isinstance(cluster, list) else cluster
        idx = []
        for c in cluster:
            assert c in self.kmeans_labels
            idx.extend(np.where(self.kmeans_labels == c)[0])
        return self[idx]

    def get_dataloader(self, type='pyg', split=None, cluster=None, batch_size=64, shuffle=False, **kwargs):
        dataset = self
        if split is not None:
            try:
                dataset = dataset.get_split_dataset(split)
            except:
                warnings.warn(f"Unable to load from split {split}")

        if cluster is not None:
            try:
                dataset = dataset.get_cluster_dataset(cluster)
            except:
                warnings.warn(f"Unable to load from cluster {cluster}")

        if type == 'feat_and_adj':
            return graphs_to_feat_and_adj_dataloader(dataset, batch_size, shuffle)
        elif type == 'pyg':
            return MultiEpochsPYGDataLoader(dataset, batch_size, shuffle)
        else:
            raise NotImplementedError(f'Unsupported type {type}')


class UnlabeledSegmentsDataset(SegmentsDataset):
    NETWORK_REPOSITORY_LINKS = {
        'ca-IMDB': 'https://nrvis.com/download/data/ca/ca-IMDB.zip',
        'ca-CondMat': 'https://nrvis.com/download/data/ca/ca-CondMat.zip',
        'ca-AstroPh': 'https://nrvis.com/download/data/ca/ca-AstroPh.zip',
        'ca-HepPh': 'https://nrvis.com/download/data/ca/ca-HepPh.zip',
        'bio-mouse-gene': 'https://nrvis.com/download/data/bio/bio-mouse-gene.zip',
        'bio-CE-CX': 'https://nrvis.com/download/data/bio/bio-CE-CX.zip',
        'bn-human-Jung2015_M87123456': 'https://nrvis.com/download/data/bn/bn-human-Jung2015_M87123456.zip',
        'econ-poli-large': 'https://nrvis.com/download/data/econ/econ-poli-large.zip',
        'email-EU': 'https://nrvis.com/download/data/email/email-EU.zip',
        'rec-movielens-tag-movies-10m': 'https://nrvis.com/download/data/rec/rec-movielens-tag-movies-10m.zip',
        'soc-epinions': 'https://nrvis.com/download/data/soc/soc-epinions.zip',
        'soc-anybeat': 'https://nrvis.com/download/data/soc/soc-anybeat.zip'
    }
    SNAP_LINKS = {
        'soc-Pokec': 'https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz',
        'roadNet-CA': 'https://snap.stanford.edu/data/roadNet-CA.txt.gz',
    }
    DATASET_DICT = {
        'network_repository': [
            'ca-CondMat', 'ca-HepPh', 'bio-CE-CX', 'email-EU', 
            'rec-movielens-tag-movies-10m', 'soc-epinions',
        ],  # bad ones: 'ca-AstroPh', 'econ-poli-large', 'bio-mouse-gene' (format)
        'large_network_repository': [
            'bn-human-Jung2015_M87123456', 
            # 'ca-IMDB' and 'bio-mouse-gene' needs to change format
        ],
        'planetoid': ['citeseer', 'pubmed', 'cora'],
        'pyg': ['QM9', 'ZINC'],
        'snap': ['soc-Pokec', 'roadNet-CA'],
    }

    def __init__(
            self, data_name_list=['network_repository'], dir_path='./data/misc', segment=True, 
            thres=1000, fill_zeros=False, max_node_num=None, return_list=False, feature_flag=False, 
            feature_types=None, embedding_dim=60, walk_length_node2vec=20, context_size=20, 
            walks_per_node=1, p_node2vec=1.0, q_node2vec=1.0, num_negative_samples=1, p_ppr=0.85, 
            beta_katz=0.005, path_len=3, remove=False, aggr='add', aggr_repeat=10, property_types=['all'],
            n_clusters=8, seed=10, scaler_dict=None, kmeans=None, **kwargs
        ):
        self.data_name = '_'.join(sorted(data_name_list))
        self.data_name_list = self.check_data_name(data_name_list)
        self.dir_path = dir_path
        self.segment = segment
        self.thres = thres
        self.fill_zeros = fill_zeros
        self.max_node_num = max_node_num
        self.return_list = return_list

        self.feature_flag = feature_flag
        if self.feature_flag and feature_types is None:
            feature_types = DEFAULT_FEATURE_TYPES
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

        self.splits = ['train'] * len(self)
    
    @property
    def processed_file_names(self):
        save_name = f"{'_'.join(sorted(self.data_name_list))}"
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

        save_name += '.pt'
        file_names = [save_name]

        split_name = f"{'_'.join(sorted(self.property_types))}"
        if self.scaler_dict is not None:
            split_name += '_pre_scale'

        if self.kmeans is not None:
            split_name += '_pre_kmeans'
        else:
            split_name += f'_{self.n_clusters}clusters'

        if self.segment and self.return_list:
            split_name += f'_seg-thres{self.thres}'

        split_name += '.pt'        
        file_names.append(split_name)

        return file_names

    def check_data_name(self, data_name_list):
        # sanity check of data_name_list, input must be list

        # if not isinstance(data_name_list, list):
        #     if data_name_list in list(self.DATASET_DICT):
        #         data_name_list = self.DATASET_DICT[data_name_list]
        #     elif data_name_list == 'all':
        #         import itertools
        #         data_name_list = list(itertools.chain.from_iterable(self.DATASET_DICT.values()))
        #     else:
        #         data_name_list = [data_name_list]
        # else:
        temp_list = []
        for d in data_name_list:
            if d in self.DATASET_DICT.keys():
                temp_list.extend(self.DATASET_DICT[d])
            else:
                temp_list.append(d)
        data_name_list = temp_list
        return data_name_list

    def get_data(self, data_name='ca-CondMat', dir_path='./data/misc', remove_isolated=True, 
                 add_self_loop=False, make_undirected=True):
        if (
            data_name in self.DATASET_DICT['network_repository'] or 
            data_name in self.DATASET_DICT['large_network_repository'] or 
            data_name in self.DATASET_DICT['snap']
        ):
            if not osp.exists(f'{dir_path}/{data_name}'):
                if data_name in self.NETWORK_REPOSITORY_LINKS:
                    url = self.NETWORK_REPOSITORY_LINKS[data_name]
                    download_unzip(url, f'{dir_path}/{data_name}')
                elif data_name in self.SNAP_LINKS:
                    url = self.NETWORK_REPOSITORY_LINKS[data_name]
                    download_file(url, f'{dir_path}/{data_name}/{data_name}.txt.gz')

            files = os.listdir(f'{dir_path}/{data_name}')
            data_file = [x for x in files if x.startswith(data_name)][0]
            try:
                from scipy.io import mmread
                
                edges_coo = mmread(f'{dir_path}/{data_name}/{data_file}')
                edge_index = torch.stack(
                    (torch.from_numpy(edges_coo.col), torch.from_numpy(edges_coo.row))
                ).long()
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

            # modification
            if remove_isolated:
                edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr)
            if add_self_loop:
                edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
            if make_undirected:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr)

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

    def process(self):
        data_list = []
        for d in self.data_name_list:
            # assert d not in self.DATASET_DICT['pyg'], f'Cannot get induced graphs of dataset {d}'
            print(f'Dataset: {d}')
            data, type = self.get_data(d, self.dir_path)

            if type == 'dataset':
                # TODO: multiple graphs
                pass
            else:
                if self.feature_flag:
                    n2v_prefix = f"-dim_{self.feat_config['embedding_dim']}" if 'node2vec' in self.feature_types else None
                    save_path = f'{self.dir_path}/{d}/{self.feat_name}{n2v_prefix}-embdedding.pt'
                    if not osp.exists(save_path):
                        feat_dict = get_features(
                            data.edge_index, data.edge_attr, data.num_nodes, self.feature_types,
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
                        data.x = aggregate_edge_attr(node_feat, data.edge_index, edge_attr, 
                                                     repeat=self.aggr_repeat, aggr=self.aggr)
                    else:
                        data.x = node_feat

                    # sanity check for data.x
                    data.x = torch.nan_to_num(data.x)

            if self.segment:
                out = extract_segments(data, thres=self.thres, fill_zeros=self.fill_zeros, return_list=self.return_list,
                                    max_node_num=self.max_node_num)
                temp_list = out if self.return_list else [out]
            else:
                data_list = [data]
            data_list.extend(temp_list)

        for g in data_list:
            # attributes for sampling
            adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0].long()
            row, col = torch.triu_indices(g.num_nodes, g.num_nodes,1)
            g.full_edge_index = torch.stack([row, col])
            g.full_edge_attr = adj[g.full_edge_index[0], g.full_edge_index[1]]
            g.nodes_per_graph = g.num_nodes
            g.edges_per_graph = g.num_nodes * (g.num_nodes - 1) // 2
            if g.edge_attr is None:
                g.edge_attr = torch.ones(g.num_edges, dtype=torch.float32)

        # calculate structure properties
        data_list, property_attr, self.property_dict, self.scaler_dict = get_properties(
            data_list, self.property_types, self.scaler_dict
        )
        self.kmeans, self.kmeans_labels = split_by_property(property_attr, self.kmeans, self.n_clusters, self.seed)
        
        self.data, self.slices = self.collate(data_list)
        print(f'Saving data to {self.processed_paths[0]}')
        torch.save((self.data, self.slices), self.processed_paths[0])
        torch.save((self.kmeans, self.kmeans_labels, self.property_dict, self.scaler_dict), 
                   self.processed_paths[1])


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--type', type=str, default='unlabeled')
    parser.add_argument('--data_name', type=str, default='ca-CondMat')
    parser.add_argument('--data_name_list', type=str, nargs='+', default='network_repository')
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
    parser.add_argument('--split_data', action='store_true')
    parser.add_argument('--get_prop', action='store_true')
    args = parser.parse_args()
    print(args)
    torch.set_num_threads(args.num_threads)

    extra_configs = {}
    if args.get_prop:
        from dataset.property import get_properties, k_means

        kmeans, _, _, scaler_dict = torch.load(
            'data/misc/large_network_repository_network_repository_snap/processed/all_8clusters_seg-thres1000.pt'
        )
        extra_configs['kmeans'] = kmeans
        extra_configs['scaler_dict'] = scaler_dict

    if args.type == 'downstream':
        dataset = DownstreamSegmentsDataset(
            task=args.task, data_name=args.data_name, segment=args.segment, thres=args.thres,
            fill_zeros=args.fill_zeros, max_node_num=args.max_node_num, return_list=args.return_list,
            split_data=args.split_data
        )
    elif args.type == 'unlabeled':
        dataset = UnlabeledSegmentsDataset(
            data_name_list=args.data_name_list, segment=args.segment, thres=args.thres,
            fill_zeros=args.fill_zeros, max_node_num=args.max_node_num, return_list=args.return_list,
            **extra_configs
        )


    num_nodes = np.array([g.num_nodes for g in dataset])
    num_edges = np.array([g.num_edges for g in dataset])
    print(f'num_graphs: {len(dataset)}')
    print(f'num_nodes: {np.mean(num_nodes)} +/- {np.std(num_nodes)}, max = {np.max(num_nodes)}, min = {np.min(num_nodes)}')
    print(f'num_edges: {np.mean(num_edges)} +/- {np.std(num_edges)}, max = {np.max(num_edges)}, min = {np.min(num_edges)}')

    import ipdb; ipdb.set_trace()