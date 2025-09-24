import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
from torch_geometric.utils import degree, contains_self_loops, remove_self_loops


DEFAULT_PROPERTY_CHOICE = ['entropy', 'deg_mean', 'deg_var', 'density', 'alpha']

# -------- extract structure properties --------
class ZScore:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, array):
        self.mean = np.mean(array)
        self.std = np.std(array)
        return self

    def transform(self, array):
        return (array - self.mean) / self.std

    def fit_transform(self, array):
        self.fit(array)
        return self.transform(array)

def compute_alpha(degree_sequence, d_min):
    """
    Approximate the alpha of a power law distribution.
    Parameters
    ----------
    degree_sequence: degree sequence
    d_min: int
        The minimum degree of nodes to consider
    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """
    S_d = torch.sum(torch.log(degree_sequence[degree_sequence >= d_min]))
    n = torch.sum(degree_sequence >= d_min)
    return n / (S_d - n * np.log(d_min - 0.5)) + 1

def compute_graph_properties(edge_index, num_nodes=None, property_types=['all']):
    if 'all' in property_types:
        property_types = DEFAULT_PROPERTY_CHOICE

    if contains_self_loops(edge_index):
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = edge_index.max().cpu().item() + 1 if num_nodes is None else num_nodes
    num_edges = edge_index.shape[1]

    property_dict = {}

    # degree mean and variance
    degrees = degree(edge_index[0], num_nodes=num_nodes)
    if 'deg_mean' in property_types:
        property_dict['deg_mean'] = degrees.mean().item()
    if 'deg_var' in property_types:
        property_dict['deg_var'] = degrees.var().item()

    # network entropy
    if 'entropy' in property_types:
        entropy = degrees.clone()
        entropy.apply_(lambda x: x * np.log(x))
        # property_dict['entropy'] = entropy.nan_to_num().sum().item() / 2 / num_edges
        try:
            property_dict['entropy'] = entropy.nan_to_num().sum().item() / num_edges
        except:
            property_dict['entropy'] = 0

    # density
    if 'density' in property_types:
        # property_dict['density'] = 2 * num_edges / num_nodes / (num_nodes - 1)
        try:
            property_dict['density'] = num_edges / num_nodes / (num_nodes - 1)
        except:
            property_dict['density'] = 0

    # scale-free exponent
    if 'alpha' in property_types:
        property_dict['alpha'] = compute_alpha(degrees, 1).item()
    
    if 'num_nodes' in property_types:
        property_dict['num_nodes'] = num_nodes
    
    if 'num_edges' in property_types:
        property_dict['num_edges'] = num_edges

    return property_dict

def get_properties(data_list, property_types=['all'], scaler_dict=None, nodes_and_edges=False,
                   attr_name=None):
    if 'all' in property_types:
        property_types = DEFAULT_PROPERTY_CHOICE

    if nodes_and_edges:
        property_types = property_types + ['num_nodes', 'num_edges']

    property_dict = {}
    for p in property_types:
        property_dict[p] = []
    
    # compute
    for data in tqdm(data_list):
        temp_dict = compute_graph_properties(data.edge_index, data.num_nodes, property_types)
        for p in property_types:
            property_dict[p].append(temp_dict[p])
    
    # make them numpy array
    for p in property_types:
        property_dict[p] = np.nan_to_num(np.array(property_dict[p]))

    # fit scalers
    if scaler_dict is None: 
        scaler_dict = {}
        for p in property_types:
            scaler_dict[p] = ZScore().fit(property_dict[p])
    
    # scale each property
    feat = []
    for p in sorted(property_dict):
        scaler = scaler_dict[p]
        feat.append(scaler.transform(property_dict[p]))

    property_attr = torch.from_numpy(np.stack(feat, axis=1)).to(torch.float32)

    for i in range(len(data_list)):
        attr_name = 'property_attr' if attr_name is None else attr_name
        data_list[i][attr_name] = property_attr[i].unsqueeze(0)

    return data_list, property_attr, property_dict, scaler_dict

# -------- cluster graphs based on structure properties --------
def k_means(X, n_clusters=None, seed=1, max_n_clusters=10):
    if n_clusters is None:
        scores = []
        for k in range(2, max_n_clusters):
            kmeans = KMeans(n_clusters=k, random_state=seed).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            scores.append(score)

        optimal_k = range(2, max_n_clusters)[scores.index(max(scores))]
        kmeans = KMeans(n_clusters=optimal_k, random_state=seed).fit(X)

    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
    
    return kmeans

def split_by_property(property_attr, kmeans=None, n_clusters=8, seed=10):
    if kmeans is None:
        kmeans = k_means(property_attr, n_clusters=n_clusters, seed=seed)
    else:
        kmeans = kmeans

    return kmeans, kmeans.predict(property_attr)