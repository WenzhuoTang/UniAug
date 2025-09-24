import random

import torch
import torch.nn.functional as F
import torch_geometric as pyg
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


# -------- difference and intersection between two tensors --------
def setdiff(t1, t2, dim=1):
    combined = torch.cat((t1, t2), dim=dim)
    uniques, counts = combined.unique(return_counts=True, dim=dim)
    difference = torch.index_select(uniques, dim, torch.where(counts == 1)[0])
    intersection = torch.index_select(uniques, dim, torch.where(counts > 1)[0])

    return difference, intersection


# -------- diffusion utils --------
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


class EmpiricalEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, degree=True, augment_features=[], **kwargs):
        # pmf of graph size
        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)

        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()

        # empty graph table
        self.empty_graphs = {}

        # degree table
        self.degree = degree
        self.augment_features = augment_features

        self.empirical_node_feat_dist = {}

        for pyg_data in train_pyg_datas:
            if pyg_data.num_nodes not in self.empirical_node_feat_dist:
                self.empirical_node_feat_dist[pyg_data.num_nodes] = []
            feats = {}
            if self.degree:
                feats['degree'] = pyg.utils.degree(pyg_data.edge_index[0],num_nodes=pyg_data.num_nodes)
            for feat_name in self.augment_features:
                feats[feat_name] = getattr(pyg_data, feat_name)# FEATURE_EXTRACTOR[feat_name]['func'](pyg_data)
            # feats['x'] = pyg_data.x
            self.empirical_node_feat_dist[pyg_data.num_nodes].append(feats)


    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        ret = ret.tolist()
        xT_feats = [] 
        for n_node in ret:
            xT_feats.append(random.choice(self.empirical_node_feat_dist[n_node]))
        # xT_feats will be a list of dicts
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)
                is_edge_indices = pyg_data.full_edge_attr.nonzero(as_tuple=True)[0]
                pyg_data.edge_index = pyg_data.full_edge_index[:, is_edge_indices]

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas

def collate_fn(pyg_datas, repeat=1):
    pyg_datas = sum([[pyg_data.clone() for _ in range(repeat)]for pyg_data in pyg_datas],[])
    batched_data = pyg.data.Batch.from_data_list(pyg_datas)
    batched_data.nodes_per_graph = torch.tensor([pyg_data.num_nodes for pyg_data in pyg_datas])
    batched_data.edges_per_graph = torch.tensor([pyg_data.num_nodes * (pyg_data.num_nodes-1)//2 for pyg_data in pyg_datas])

    return batched_data 