import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch_geometric
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch


# -------- manually construct  --------
def get_batched_datalist(dataset, nodes_max=None, edges_max=None):
    if nodes_max is not None or edges_max is not None:
        thres = []
        counts = []
        if nodes_max is not None:
            thres.append(nodes_max)
            counts.append(np.array([g.num_nodes for g in dataset]))
        if edges_max is not None:
            thres.append(edges_max)
            counts.append(np.array([g.num_edges for g in dataset]))

        order = np.argsort(np.array([g.num_edges for g in dataset]))
        counts_cumsum = [np.cumsum(counts[i][order]) for i in range(len(counts))]

        ptr_list = []
        while True:
            idx_list = [np.where(counts_cumsum[i] > thres[i])[0] for i in range(len(counts))]

            temp_ptr = [idx[0] for idx in idx_list if len(idx) > 0]
            if len(temp_ptr) > 0:
                ptr = min(temp_ptr)

                if ptr == 0 or (len(ptr_list) > 0 and ptr == ptr_list[-1]):
                    ptr += 1

                if ptr >= len(dataset):
                    break

                ptr_list.append(ptr)
                counts_cumsum = [cnt_cs - cnt_cs[ptr - 1] for cnt_cs in counts_cumsum]
            else:
                break

        ptr = np.array(ptr_list)
        splits = np.split(np.arange(len(dataset)), ptr)

        mapping = dict(zip(np.argsort(order).tolist(), range(len(dataset))))
        order2idx = np.vectorize(lambda x: mapping.__getitem__(x))
        splits = [order2idx(s) for s in splits]

        data_list = []
        for s in splits:
            data_list.append(Batch.from_data_list([dataset[i] for i in s]))

    else:
        data_list = dataset
    
    return data_list


# -------- fast multi epochs dataloader --------
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = RepeatSampler(self.sampler)
        else:
            self.batch_sampler = RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class MultiEpochsPYGDataLoader(torch_geometric.loader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = RepeatSampler(self.sampler)
        else:
            self.batch_sampler = RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def graphs_to_feat_and_adj_dataloader(graph_list, batch_size=1, shuffle=False, use_edge_attr=False):
    adjs_list = []
    x_list = []
    for g in graph_list:
        edge_attr = g.edge_attr if use_edge_attr else None
        adjs_list.append(
            to_dense_adj(
                g.edge_index, edge_attr=edge_attr, max_num_nodes=g.num_nodes
            )[0]
        )
        x_list.append(g.x)
    
    adjs_tensor = torch.stack(adjs_list).float()
    x_tensor = torch.stack(x_list).float()

    dataset = TensorDataset(x_tensor, adjs_tensor)
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def graphs_to_pt_dataloader(graph_list, batch_size=1, shuffle=False):
    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)


# -------- utils for NCN iterator --------
class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret