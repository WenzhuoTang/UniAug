import os
import dgl
import time
import pyemd
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import torch
import torch_geometric
from torch_geometric.utils import to_networkx

from functools import partial
from eden.graph import vectorize
from scipy.linalg import toeplitz, eigvalsh, sqrtm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

from eval.gin import load_feature_extractor


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper


class GenericGraphEvaluator:
    def __init__(self, reference_datalist, exp_name='run', device='cuda:0', use_pretrained=False, model_path=None):
        self.device = device
        model = load_feature_extractor(device=device, use_pretrained=use_pretrained, model_path=model_path)
        self.references = [to_networkx(g.cpu(), to_undirected=True) for g in reference_datalist]
        self.evaluators = {
            'degree': MMDEval(statistic='degree'), 
            'spectral': MMDEval(statistic='spectral'),
            'clustering': MMDEval(statistic='clustering'),
            'orbits': MMDEval(statistic='orbits', exp_name=exp_name),
            'fid': FIDEvaluation(model=model)
        }

    def __call__(self, target_datalist):
        targets = [to_networkx(g.cpu(), to_undirected=True) for g in target_datalist]
        metrics = {}
        metrics['degree_mmd'] = self.evaluators['degree'].evaluate(targets, self.references)[0]['degree_mmd']
        metrics['spectral_mmd'] = self.evaluators['spectral'].evaluate(targets, self.references)[0]['spectral_mmd']
        metrics['clustering_mmd'] = self.evaluators['clustering'].evaluate(targets, self.references)[0]['clustering_mmd']
        metrics['orbits_mmd'] = self.evaluators['orbits'].evaluate(targets, self.references)[0]['orbits_mmd']

        targets_dgl = [dgl.DGLGraph(g).to(self.device) for g in targets]
        references_dgl = [dgl.DGLGraph(g).to(self.device) for g in self.references]
        metrics['fid'] = self.evaluators['fid'].evaluate(targets_dgl, references_dgl)[0]['fid']
        return metrics


class GINMetric():
    def __init__(self, model):
        self.feat_extractor = model
        self.get_activations = self.get_activations_gin

    @time_function
    def get_activations_gin(self, generated_dataset, reference_dataset):
        return self._get_activations(generated_dataset, reference_dataset)

    def _get_activations(self, generated_dataset, reference_dataset):
        gen_activations = self.__get_activations_single_dataset(generated_dataset)
        ref_activations = self.__get_activations_single_dataset(reference_dataset)

        scaler = StandardScaler()
        scaler.fit(ref_activations)
        ref_activations = scaler.transform(ref_activations)
        gen_activations = scaler.transform(gen_activations)

        return gen_activations, ref_activations

    def __get_activations_single_dataset(self, dataset):
        node_feat_loc = self.feat_extractor.node_feat_loc
        edge_feat_loc = self.feat_extractor.edge_feat_loc

        ndata = [node_feat_loc] if node_feat_loc in dataset[0].ndata\
            else '__ALL__'
        edata = [edge_feat_loc] if edge_feat_loc in dataset[0].edata\
            else '__ALL__'
        graphs = dgl.batch(
            dataset, ndata=ndata, edata=edata).to(self.feat_extractor.device)

        if node_feat_loc not in graphs.ndata:  # Use degree as features
            feats = graphs.in_degrees() + graphs.out_degrees()
            feats = feats.unsqueeze(1).type(torch.float32)
        else:
            feats = graphs.ndata[node_feat_loc]
        feats = feats.to(self.feat_extractor.device)

        graph_embeds = self.feat_extractor(graphs, feats)
        return graph_embeds.cpu().detach().numpy()

    def evaluate(self, *args, **kwargs):
        raise Exception('Must be implemented by child class')


class FIDEvaluation(GINMetric):
    # https://github.com/mseitzer/pytorch-fid
    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        mu_ref, cov_ref = self.__calculate_dataset_stats(reference_dataset)
        mu_generated, cov_generated = self.__calculate_dataset_stats(generated_dataset)
        # print(np.max(mu_generated), np.max(cov_generated), 'mu, cov fid')
        fid = self.compute_FID(mu_ref, mu_generated, cov_ref, cov_generated)
        return {'fid': fid}

    def __calculate_dataset_stats(self, activations):
        # print('activation mean -----------------------------------------', activations.mean())
        mu = np.mean(activations, axis = 0)
        cov = np.cov(activations, rowvar = False)

        return mu, cov

    def compute_FID(self, mu1, mu2, cov1, cov2, eps = 1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = sqrtm(cov1.dot(cov2), disp=False)
        # print(np.max(covmean), 'covmean')
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # print(tr_covmean, 'tr_covmean')

        return (diff.dot(diff) + np.trace(cov1) +
                np.trace(cov2) - 2 * tr_covmean)


class MMDEval:
    # Largely taken from the GraphRNN github: https://github.com/JiaxuanYou/graph-generation
    # I just rearranged to make it a little cleaner.
    def __init__(self, statistic='degree', **kwargs):
        if statistic == 'degree':
            self.descriptor = Degree(**kwargs)
        elif statistic == 'clustering':
            self.descriptor = Clustering(**kwargs)
        elif statistic == 'orbits':
            self.descriptor = Orbits(**kwargs)
        elif statistic == 'spectral':
            self.descriptor = Spectral(**kwargs)
        else:
            raise Exception('unsupported statistic'.format(statistic))


    def evaluate(self, generated_dataset=None, reference_dataset=None):
        reference_dataset = self.extract_dataset(reference_dataset)
        generated_dataset = self.extract_dataset(generated_dataset)
        if len(reference_dataset) == 0 or len(generated_dataset) == 0:
            return {f'{self.descriptor.name}_mmd': 0}, 0

        start = time.time()
        metric = self.descriptor.evaluate(generated_dataset, reference_dataset)
        total = time.time() - start
        return {f'{self.descriptor.name}_mmd': metric}, total


    def extract_dataset(self, dataset):
        if isinstance(dataset[0], nx.Graph):
            pass
        elif isinstance(dataset[0], torch_geometric.data.Data):
            dataset = [to_networkx(g.cpu(), to_undirected=True) for g in dataset]
        else:
            raise Exception(f'Unsupported element type {type(dataset[0])} for dataset, \
                expected list of nx.Graph or dgl.DGLGraph')

        return [g for g in dataset if g.number_of_nodes() != 0]


class Descriptor:
    def __init__(self, is_parallel=True, bins=100, kernel='gaussian_emd', max_workers=32,
                 sigma_type='single', **kwargs):
        self.is_parallel = is_parallel
        self.bins = bins
        self.max_workers = max_workers

        if kernel == 'gaussian_emd':
            self.distance = self.emd
        elif kernel == 'gaussian_rbf':
            self.distance = self.l2
            self.name += '_rbf'
        else:
            raise Exception(kernel)

        if sigma_type == 'range':
            self.name += '_range'
            self.sigmas += [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
            self.__get_sigma_mult_factor = self.mean_pairwise_distance
        else:
            self.__get_sigma_mult_factor = self.identity

        self.sigmas = np.array(list(set(self.sigmas)))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.sigmas * mult_factor

    def mean_pairwise_distance(self, GR):
        return np.sqrt(GR.mean())

    def identity(self, *args, **kwargs):
        return 1

    def evaluate(self, generated_dataset, reference_dataset):
        ''' Compute the distance between the distributions of two unordered sets of graphs.
        Args:
          graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
        sample_pred = self.extract_features(generated_dataset)
        sample_ref = self.extract_features(reference_dataset)

        GG = self.disc(sample_pred, sample_pred, distance_scaling=self.distance_scaling)
        GR = self.disc(sample_pred, sample_ref, distance_scaling=self.distance_scaling)
        RR = self.disc(sample_ref, sample_ref, distance_scaling=self.distance_scaling)

        sigmas = self.get_sigmas(GR)
        max_mmd = 0
        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - (2 * K_GR.mean())
            max_mmd = mmd if mmd > max_mmd else max_mmd
            # print(mmd, max_mmd)

        return max_mmd

    def pad_histogram(self, x, y):
        support_size = max(len(x), len(y))
        # convert histogram values x and y to float, and make them equal len
        x = x.astype(float)
        y = y.astype(float)
        if len(x) < len(y):
            x = np.hstack((x, [0.0] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.0] * (support_size - len(y))))

        return x, y

    def emd(self, x, y, distance_scaling=1.0):
        support_size = max(len(x), len(y))
        x, y = self.pad_histogram(x, y)

        d_mat = toeplitz(range(support_size)).astype(float)
        distance_mat = d_mat / distance_scaling

        dist = pyemd.emd(x, y, distance_mat)
        return dist ** 2

    def l2(self, x, y, **kwargs):
        x, y = self.pad_histogram(x, y)
        dist = np.linalg.norm(x - y, 2)
        return dist ** 2

    def gaussian_tv(self, x, y): #, sigma=1.0, *args, **kwargs):
        x, y = self.pad_histogram(x, y)

        dist = np.abs(x - y).sum() / 2.0
        return dist ** 2

    def kernel_parallel_unpacked(self, x, samples2, kernel):
        dist = []
        for s2 in samples2:
            dist += [kernel(x, s2)]
        return dist

    def kernel_parallel_worker(self, t):
        return self.kernel_parallel_unpacked(*t)

    def disc(self, samples1, samples2, **kwargs):
        ''' Discrepancy between 2 samples
        '''
        tot_dist = []
        if not self.is_parallel:
            for s1 in samples1:
                for s2 in samples2:
                    tot_dist += [self.distance(s1, s2)]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for dist in executor.map(self.kernel_parallel_worker,
                        [(s1, samples2, partial(self.distance, **kwargs)) for s1 in samples1]):
                    tot_dist += [dist]
        return np.array(tot_dist)


class Degree(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'degree'
        self.sigmas = [1.0]
        self.distance_scaling = 1.0
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for deg_hist in executor.map(self.degree_worker, dataset):
                    res.append(deg_hist)
        else:
            for g in dataset:
                degree_hist = self.degree_worker(g)
                res.append(degree_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def degree_worker(self, G):
        return np.array(nx.degree_histogram(G))


class Clustering(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'clustering'
        self.sigmas = [1.0 / 10]
        super().__init__(*args, **kwargs)
        self.distance_scaling = self.bins


    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for clustering_hist in executor.map(self.clustering_worker,
                    [(G, self.bins) for G in dataset]):
                    res.append(clustering_hist)
        else:
            for g in dataset:
                clustering_hist = self.clustering_worker((g, self.bins))
                res.append(clustering_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def clustering_worker(self, param):
        G, bins = param
        clustering_coeffs_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        return hist


class Orbits(Descriptor):
    motif_to_indices = {
            '3path' : [1, 2],
            '4cycle' : [8],
    }
    COUNT_START_STR = 'orbit counts: \n'
    def __init__(self, exp_name='run', *args, **kwargs):
        self.exp_name = exp_name
        self.name = 'orbits'
        self.sigmas = [30.0]
        self.distance_scaling = 1
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        for G in dataset:
            try:
                orbit_counts = self.orca(G)
            except:
                continue
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
            res.append(orbit_counts_graph)
        return np.array(res)

    def orca(self, graph):
        root = './dataset/orca'
        # ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        # tmp_fname = f'{root}/{self.exp_name}-{ts}.txt'
        tmp_fname = f'{root}/{self.exp_name}.txt'
        f = open(tmp_fname, 'w')
        f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
        for (u, v) in self.edge_list_reindexed(graph):
            f.write(str(u) + ' ' + str(v) + '\n')
        f.close()

        output = sp.check_output([f'{root}/orca', 'node', '4', tmp_fname, 'std'])
        output = output.decode('utf8').strip()

        idx = output.find(self.COUNT_START_STR) + len(self.COUNT_START_STR)
        output = output[idx:]
        node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
            for node_cnts in output.strip('\n').split('\n')])

        try:
            os.remove(tmp_fname)
        except OSError:
            pass

        return node_orbit_counts

    def edge_list_reindexed(self, G):
        idx = 0
        id2idx = dict()
        for u in G.nodes():
            id2idx[str(u)] = idx
            idx += 1

        edges = []
        for (u, v) in G.edges():
            edges.append((id2idx[str(u)], id2idx[str(v)]))
        return edges


class Spectral(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'spectral'
        self.sigmas = [1.0]
        self.distance_scaling = 1
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for spectral_density in executor.map(self.spectral_worker, dataset):
                    res.append(spectral_density)
        else:
            for g in dataset:
                spectral_temp = self.spectral_worker(g)
                res.append(spectral_temp)
        return res

    def spectral_worker(self, G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        spectral_pmf = spectral_pmf / spectral_pmf.sum()
        return spectral_pmf


# TODO: accomodate pyg dataset
class NSPDKEvaluation():
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        # prepare - dont include in timing
        generated_dataset_nx = [to_networkx(g.cpu(), to_undirected=True) for g in generated_dataset if g.num_nodes != 0]
        reference_dataset_nx = [to_networkx(g.cpu(), to_undirected=True) for g in reference_dataset if g.num_nodes != 0]

        if len(reference_dataset_nx) == 0 or len(generated_dataset_nx) == 0:
            return {'nspdk_mmd': 0}, 0

        if 'attr' not in generated_dataset[0].ndata:
            [nx.set_node_attributes(g, {key: str(val) for key, val in dict(g.degree()).items()}, 'label') for g in generated_dataset_nx]  # degree labels
            [nx.set_node_attributes(g, {key: str(val) for key, val in dict(g.degree()).items()}, 'label') for g in reference_dataset_nx]  # degree labels
            [nx.set_edge_attributes(g, '1', 'label') for g in generated_dataset_nx]  # degree labels
            [nx.set_edge_attributes(g, '1', 'label') for g in reference_dataset_nx]  # degree labels

        else:
            self.set_features(generated_dataset, generated_dataset_nx)
            self.set_features(reference_dataset, reference_dataset_nx)

        return self.evaluate_(generated_dataset_nx, reference_dataset_nx)

    def set_features(self, dset_dgl, dset_nx):
        for g_dgl, g_nx in zip(dset_dgl, dset_nx):
            feat_dict = {node: str(g_dgl.ndata['attr'][node].nonzero().item()) for node in range(g_dgl.number_of_nodes())}
            nx.set_node_attributes(g_nx, feat_dict, 'label')

            srcs, dests, eids = g_dgl.edges('all')
            feat_dict = {}
            for src, dest, eid in zip(srcs, dests, eids):
                feat_dict[(src.item(), dest.item())] = str(g_dgl.edata['attr'][eid].nonzero().item())
                # feat_dict = {edge: g.edata['attr'][edge].nonzero() for edge in range(g.number_of_edges())}
            # print(feat_dict)
            nx.set_edge_attributes(g_nx, feat_dict, 'label')

    @time_function
    def evaluate_(self, generated_dataset, reference_dataset):
        ref = vectorize(reference_dataset, complexity=4, discrete=True)
        for g in reference_dataset:
            del g

        gen = vectorize(generated_dataset, complexity=4, discrete=True)
        for g in generated_dataset:
            del g

        K_RR = pairwise_kernels(ref, ref, metric='linear', n_jobs=4)
        K_GG = pairwise_kernels(gen, gen, metric='linear', n_jobs=4)
        K_GR = pairwise_kernels(ref, gen, metric='linear', n_jobs=4)

        mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()

        return {'nspdk_mmd': mmd}