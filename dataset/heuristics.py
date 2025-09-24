

import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader


def CN(A, edge_index, batch_size=100000, reshape=True):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores


def AA(A, edge_index, batch_size=100000, reshape=True):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores

def RA(A, edge_index, batch_size=100000, reshape=True):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores

def PPR(A, edge_index, p=0.85, reshape=True):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    # p: damping factor
    from fast_pagerank import pagerank_power

    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    j = 0
    for i in range(edge_index.shape[1]):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=p, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores


def shortest_path(A, edge_index, remove=False, reshape=True):
    
    scores = []
    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_scipy_sparse_array(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        # if (s,t) in train_pos_list: train_pos_list.remove((s,t))
        # if (t,s) in train_pos_list: train_pos_list.remove((t,s))


        # G = nx.Graph(train_pos_list)
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):

            sp = nx.shortest_path_length(G, source=s, target=t)
            # if sp == 0:
            #     print(1)
        else:
            sp = 999
        

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    

        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    scores = torch.FloatTensor(scores)
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False, reshape=True):
    scores = []
    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_scipy_sparse_array(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    scores = torch.FloatTensor(scores)
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores


def katz_close(A, edge_index, beta=0.005, reshape=True):

    scores = []
    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_scipy_sparse_array(A)

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])

    
    scores = torch.FloatTensor(scores)
    if reshape:
        scores = scores.reshape(-1, 1)
    return scores
