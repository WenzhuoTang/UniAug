import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch_scatter import scatter
from torch_geometric.utils import negative_sampling

from diffusion.diffusion_utils import (
    log_1_min_a, log_add_exp, extract, setdiff,
    log_categorical, index_to_log_onehot,
)
from diffusion.noise_schedule import binary_alpha_schedule
from dataset.loader import MultiEpochsDataLoader
from dataset.feature import get_features
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


class BinaryDiffusion(torch.nn.Module):
    def __init__(self, model, initial_graph_sampler=None, num_timesteps=1000, loss_type='vb_kl', pi=1e-12,
                 parametrization='x0', sample_time_method='importance', schedule_type='linear', **kwargs):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.num_edge_classes = 2
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.sample_time_method = sample_time_method
        self.initial_graph_sampler = initial_graph_sampler
        final_prob_edge = [1 - pi, pi]
        log_final_prob_edge = torch.tensor(final_prob_edge)[None, :].log()

        alphas = binary_alpha_schedule(num_timesteps, schedule_type)
        alphas = torch.tensor(alphas, dtype=float)
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_final_prob_edge', log_final_prob_edge.float())

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
    
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    
    def q_sample(self, batched_graph, t_edge):
        log_prob_edge = self._q_pred(batched_graph, t_edge)
        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)
        return log_out_edge
    
    @torch.no_grad()
    def p_sample(self, batched_graph, t_node, t_edge, full_edge_attr=None, thres=None):
        # p_sample is always one step prediction!
        log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge, full_edge_attr)
        log_out_edge = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes, thres)
        return log_out_edge
    
    def log_sample_categorical(self, logits, num_classes, thres=None):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        if thres is None:
            sample = (gumbel_noise + logits).argmax(dim=1)
        else:
            # Currently only support 2 class
            assert num_classes == 2
            sample = ((gumbel_noise + logits)[:, 1] > np.log(thres)).long()
        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample

    def log_prob(self, batched_graph):
        if self.training:
            return self._train_loss(batched_graph)
        else:
            return self._eval_loss(batched_graph)

    def _q_posterior(self, log_x_start, log_x_t, t, log_final_prob):
        assert log_x_start.shape[1] == 2, f'num_class > 2 not supported'

        tmin1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)
        log_p1 = log_final_prob[:,1]
        log_1_min_p1 = log_final_prob[:,0]

        log_x_start_real = log_x_start[:,1]
        log_1_min_x_start_real = log_x_start[:,0]

        log_x_t_real = log_x_t[:,1]
        log_1_min_x_t_real = log_x_t[:,0]

        log_alpha_t = extract(self.log_alpha, t, log_x_start_real.shape)
        log_beta_t = extract(self.log_1_min_alpha, t, log_x_start_real.shape)
        
        log_cumprod_alpha_tmin1 = extract(self.log_cumprod_alpha, tmin1, log_x_start_real.shape)
        log_1_min_cumprod_alpha_tmin1 = extract(self.log_1_min_cumprod_alpha, tmin1, log_x_start_real.shape)


        log_xtmin1_eq_0_given_x_t = log_add_exp(log_beta_t+log_p1+log_x_t_real, log_1_min_a(log_beta_t+log_p1)+log_1_min_x_t_real)
        log_xtmin1_eq_1_given_x_t = log_add_exp(log_add_exp(log_alpha_t, log_beta_t+log_p1) + log_x_t_real,
                    log_beta_t + log_1_min_p1 + log_1_min_x_t_real)

        log_xtmin1_eq_0_given_x_start = log_add_exp(log_1_min_cumprod_alpha_tmin1+log_1_min_p1, log_cumprod_alpha_tmin1 + log_1_min_x_start_real)
        log_xtmin1_eq_1_given_x_start = log_add_exp(log_cumprod_alpha_tmin1 + log_x_start_real, log_1_min_cumprod_alpha_tmin1+log_p1)

        log_xt_eq_0_given_xt_x_start = log_xtmin1_eq_0_given_x_t + log_xtmin1_eq_0_given_x_start
        log_xt_eq_1_given_xt_x_start = log_xtmin1_eq_1_given_x_t + log_xtmin1_eq_1_given_x_start

        unnorm_log_probs = torch.stack([log_xt_eq_0_given_xt_x_start, log_xt_eq_1_given_xt_x_start], dim=1)
        log_EV_xtmin_given_xt_given_xstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True)
        return log_EV_xtmin_given_xt_given_xstart

    def setup_model_input(self, batched_graph):
        # get edge_index
        edge_attr_t = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)
        batched_graph.edge_index = edge_index

        return batched_graph

    def _predict_x0_or_xtmin1(self, batched_graph, t_node):
        # if batch_size is not None:
        #     device = batched_graph.full_edge_index.device
        #     dataset = TensorDataset(batched_graph.full_edge_index.T.cpu())
        #     dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        #     out_edge_list = []
        #     for _, (b_index) in enumerate(dataloader):
        #         b_index = b_index[0].to(device)
        #         src, dst = b_index.T
        #         out_edge_list.append(self.model(batched_graph, t_node, src, dst)[0])
        #     out_edge = torch.cat(out_edge_list)
        # else:
        #     out_edge = self.model(batched_graph, t_node)[0]

        batched_graph = self.setup_model_input(batched_graph)
        out_edge = self.model(batched_graph, t_node)[0]
        assert out_edge.size(1) == self.num_edge_classes
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_edge

    def _ce_prior(self, batched_graph):
        device = batched_graph.edge_index.device
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=device).long()

        log_qxT_prob_edge = self._q_pred(batched_graph, t_edge=(self.num_timesteps - 1) * ones_edge)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)

        ce_prior_edge = -log_categorical(log_qxT_prob_edge, log_final_prob_edge)
        ce_prior_edge = scatter(
            ce_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum'
        )

        return ce_prior_edge

    def _kl_prior(self, batched_graph):
        device = batched_graph.edge_index.device
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=device).long()

        log_qxT_prob_edge = self._q_pred(batched_graph, t_edge=(self.num_timesteps - 1) * ones_edge)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)

        kl_prior_edge = self.multinomial_kl(log_qxT_prob_edge, log_final_prob_edge)
        kl_prior_edge = scatter(
            kl_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum'
        )

        return kl_prior_edge
   
    def _compute_MC_KL(self, batched_graph, t_node, t_edge):
        log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)
        log_true_prob_edge = self._q_pred_one_timestep(batched_graph=batched_graph, t_edge=t_edge)

        cross_ent_edge = -log_categorical(batched_graph.log_full_edge_attr_tmin1, log_model_prob_edge)
        ent_edge = log_categorical(batched_graph.log_full_edge_attr_t, log_true_prob_edge).detach()
        
        loss_edge = cross_ent_edge + ent_edge
        loss_edge = scatter(
            loss_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum'
        )
        return loss_edge

    def _compute_RB_KL(self, batched_graph, t, t_node, t_edge):
        log_true_prob_edge = self._q_posterior(
            log_x_start=batched_graph.log_full_edge_attr, log_x_t=batched_graph.log_full_edge_attr_t, 
            t=t_edge, log_final_prob=self.log_final_prob_edge
        )
        log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        kl_edge = self.multinomial_kl(log_true_prob_edge, log_model_prob_edge)
        kl_edge = scatter(
            kl_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum'
        )

        decoder_nll_edge = -log_categorical(batched_graph.log_full_edge_attr, log_model_prob_edge)
        decoder_nll_edge = scatter(
            decoder_nll_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum'
        )

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll_edge + (1. - mask) * kl_edge

        return loss

    def _q_sample_and_set_xt_given_x0(self, batched_graph, t_edge):
        batched_graph.log_full_edge_attr = index_to_log_onehot(
            batched_graph.full_edge_attr, self.num_edge_classes
        )
        log_full_edge_attr_t = self.q_sample(batched_graph, t_edge)
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t 

    def _q_sample_and_set_xtmin1_xt_given_x0(self, batched_graph, t_edge):
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        # sample xt-1
        tmin1_edge = t_edge - 1
        tmin1_edge_clamped = torch.where(tmin1_edge < 0, torch.zeros_like(tmin1_edge), tmin1_edge)
        
        log_full_edge_attr_tmin1 = self.q_sample(batched_graph, tmin1_edge_clamped)
        batched_graph.log_full_edge_attr_tmin1 = log_full_edge_attr_tmin1
        batched_graph.log_full_edge_attr_tmin1[tmin1_edge<0] = batched_graph.log_full_edge_attr[tmin1_edge<0]

        # sample xt given xt-1
        log_full_edge_attr_t = self._q_sample_one_timestep(batched_graph, t_edge)
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t

    # xt ~ q(xt|xtmin1)
    def _q_sample_one_timestep(self, batched_graph, t_edge):
        log_prob_edge = self._q_pred_one_timestep(batched_graph, t_edge)
        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)
        return log_out_edge  

    def _q_pred_one_timestep(self, batched_graph, t_edge):
        log_alpha_t_edge = extract(self.log_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_alpha_t_edge = extract(self.log_1_min_alpha, t_edge, batched_graph.log_full_edge_attr.shape)

        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr_tmin1 + log_alpha_t_edge,
            log_1_min_alpha_t_edge + self.log_final_prob_edge
        )

        return log_prob_edges 

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0]

    def _sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self._sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            length = self.Lt_count.shape[0]
            t = torch.randint(0, length, (b,), device=device).long()

            pt = torch.ones_like(t).float() / length
            return t, pt
        else:
            raise ValueError 

    def _q_pred(self, batched_graph, t_edge):
        # edges prob
        log_cumprod_alpha_t_edge = extract(self.log_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_cumprod_alpha_edge = extract(self.log_1_min_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr + log_cumprod_alpha_t_edge,
            log_1_min_cumprod_alpha_edge + self.log_final_prob_edge
        )
        return log_prob_edges 

    def _p_pred(self, batched_graph, t_node, t_edge, full_edge_attr=None):
        if self.parametrization == 'x0':
            log_full_edge_recon = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node)
            if full_edge_attr is not None:  # inpaint
                is_edge_indices = full_edge_attr.nonzero(as_tuple=True)[0]
                log_full_edge_attr = index_to_log_onehot(full_edge_attr, self.num_edge_classes)
                log_full_edge_recon[is_edge_indices] = log_full_edge_attr[is_edge_indices]

            log_model_pred_edge = self._q_posterior(
                log_x_start=log_full_edge_recon, log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge
            )
        elif self.parametrization == 'xt':
            log_model_pred_edge = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node)
        return log_model_pred_edge

    def _prepare_data_for_sampling(self, batched_graph):
        device = batched_graph.edge_index.device
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        log_prob_edge = torch.ones_like(
            batched_graph.log_full_edge_attr, device=device
        ) * self.log_final_prob_edge
        batched_graph.log_full_edge_attr_t = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return batched_graph

    def _eval_loss(self, batched_graph):
        b = batched_graph.num_graphs
        device = batched_graph.edge_index.device
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        t, pt = self._sample_time(b, device, self.sample_time_method)
        t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
        if self.loss_type == 'vb_kl':
            self._q_sample_and_set_xt_given_x0(batched_graph, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_node, t_edge)
            kl_prior = self._kl_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return -vb_loss

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_edge)

            kl = self._compute_MC_KL(batched_graph, t_node, t_edge)
            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior
            return -vb_loss

        else:
            raise ValueError()

    def _train_loss(self, batched_graph):
        b = batched_graph.num_graphs
        device = batched_graph.edge_index.device
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        t, pt = self._sample_time(b, device, self.sample_time_method)
        t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
        if self.loss_type == 'vb_kl':
            assert self.parametrization == 'x0'
            self._q_sample_and_set_xt_given_x0(batched_graph, t_edge)
            kl = self._compute_RB_KL(batched_graph, t, t_node, t_edge)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self._kl_prior(batched_graph=batched_graph)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return -vb_loss

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'            
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_edge)
            kl = self._compute_MC_KL(batched_graph, t_node, t_edge)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior

            return -vb_loss

        else:
            raise ValueError()
    
    def forward(self, batched_graph):
        """ Compute the log-likelihood """
        return -self.log_prob(batched_graph).sum() / (math.log(2) * batched_graph.num_entries)

    def sample(self, batched_graph=None, num_samples=10, inpaint_flag=False, device='cuda:0', 
               thres=None, num_sample_steps=None, **kwargs):
        self.model.eval()

        sampling_steps = np.arange(self.num_timesteps)
        num_sample_steps = self.num_timesteps if num_sample_steps is None else num_sample_steps
        num_sample_steps = int(num_sample_steps)
        if num_sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, num_sample_steps)
            idx = np.array(idx * (self.num_timesteps-1), int)
            sampling_steps = sampling_steps[idx]

        full_edge_attr = None
        if batched_graph is None:
            batched_graph = self.initial_graph_sampler.sample(num_samples)
            batched_graph.to(device)
            batched_graph = self._prepare_data_for_sampling(batched_graph)
            num_edges = batched_graph.edges_per_graph.sum()
        else:
            batched_graph.to(device)
            try:
                num_edges = batched_graph.edges_per_graph.sum()
            except:
                num_edges = batched_graph.edges_per_graph
            t_edge = torch.full((num_edges,), self.num_timesteps - 1, device=device, dtype=torch.long)
            self._q_sample_and_set_xt_given_x0(batched_graph, t_edge)
            if inpaint_flag:
                full_edge_attr = batched_graph.full_edge_attr
 
        try:
            num_nodes = batched_graph.nodes_per_graph.sum()
        except:
            num_nodes = batched_graph.nodes_per_graph

        # for t in tqdm.tqdm(reversed(range(0, self.num_timesteps))):
        for t in reversed(sampling_steps):
            # print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=device, dtype=torch.long)

            log_full_edge_attr_tmin1 = self.p_sample(batched_graph, t_node, t_edge, full_edge_attr, thres)
            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            torch.cuda.empty_cache()

        full_edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        if inpaint_flag:
            inpaint_indices = batched_graph.full_edge_attr.nonzero(as_tuple=True)[0]
            full_edge_attr[inpaint_indices] = batched_graph.full_edge_attr[inpaint_indices]

        is_edge_indices = full_edge_attr.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)
        batched_graph.edge_index = edge_index 

        edge_attr = full_edge_attr[is_edge_indices]
        batched_graph.edge_attr = edge_attr

        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']

        return batched_graph

class BinaryDiffusionGuidance(BinaryDiffusion):
    # XXX: Modified from https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py
    # XXX: currently only suppoorts uniform time sampling

    def __init__(self, model, guidance_head, guidance_type='link_pred', initial_graph_sampler=None, 
                 num_timesteps=128, loss_type='vb_kl', pi=1e-12, parametrization='x0', sample_time_method='uniform', 
                 schedule_type='linear', freeze_model=True, label_smoothing=True, **kwargs):
        super().__init__(
            model=model, initial_graph_sampler=initial_graph_sampler, num_timesteps=num_timesteps, 
            loss_type=loss_type, pi=pi, parametrization=parametrization, sample_time_method='uniform', 
            schedule_type=schedule_type
        )
        self.guidance_type = guidance_type
        self.guidance_mod, self.guidance_tar = self.guidance_type.split('_')
        self.guidance_head = guidance_head
        self.label_smoothing = label_smoothing

        self.freeze_model = freeze_model
        if self.freeze_model:
            self.freeze_for_discriminative()

    def freeze_for_discriminative(self):
        for _, p in enumerate(self.model.parameters()):
            p.requires_grad_(False)

        for _, p in enumerate(self.guidance_head.parameters()):
            p.requires_grad_(True)

    def freeze_for_sample(self):
        for _, p in enumerate(self.model.parameters()):
            p.requires_grad_(False)

        for _, p in enumerate(self.guidance_head.parameters()):
            p.requires_grad_(False)

    def get_corrupted_labels(self, y, t):
        # y should be one hot
        num_class = y.shape[1]
        cumprod_alpha = torch.exp(extract(self.log_cumprod_alpha, t, y.shape))
        y = cumprod_alpha * y + (1 - cumprod_alpha) / num_class
        return y

    def guidance_score(self, h, batched_graph, pos_edges=None, **kwargs):
        if self.guidance_mod == 'link':
            return self.guidance_head(h, pos_edges[0], pos_edges[1])

        elif self.guidance_mod == 'node':
            return self.guidance_head(h)

        elif self.guidance_mod == 'edge':
            edge_index = batched_graph.edge_index

            if edge_index.shape[1] > 0:
                src, dst = edge_index
                return self.guidance_head(h, src, dst)

            else:
                return torch.FloatTensor([0]).to(h.device)

        elif self.guidance_mod == 'graph':
            return self.guidance_head(h, batched_graph.batch)

        else:
            raise ValueError()

    def guidance_loss(self, h, batched_graph, pos_edges=None, t=None, **kwargs):
        if self.guidance_mod == 'link':
            assert self.guidance_tar == 'pred'
            edge_index = batched_graph.edge_index
            if edge_index.shape[1] > 0:
                # remove edge_index from pos_edges
                # diff, _ = setdiff(edge_index, pos_edges, dim=1)
                # _, pos_edges = setdiff(diff, pos_edges, dim=1)

                # pos_out = self.guidance_head(h, pos_edges[0], pos_edges[1])
                pos_out = self.guidance_head(h, edge_index[0], edge_index[1])
                pos_loss = -torch.log(pos_out + 1e-15)

                neg_edges = negative_sampling(
                    pos_edges, num_nodes=h.size(0),
                    num_neg_samples=edge_index.size(1), method='dense'
                )
                # neg_edges = torch.randint(0, batched_graph.num_nodes, pos_edges.size(),
                #                         dtype=torch.long, device=h.device)
                neg_out = self.guidance_head(h, neg_edges[0], neg_edges[1])
                neg_loss = -torch.log(1 - neg_out + 1e-15)

                loss = pos_loss + neg_loss
                loss = torch.nan_to_num(loss)
            
            else:
                loss = torch.FloatTensor([0]).to(h.device)
                loss.requires_grad_()

        elif self.guidance_mod == 'node':
            if self.guidance_tar == 'class':
                labels = batched_graph.labels
                labels = self.get_corrupted_labels(labels, t)

                logits = self.guidance_head(h, batch)
                loss = F.cross_entropy(logits, labels)

            else:
                labels = get_features(
                    batched_graph.edge_index.cpu(), num_nodes=batched_graph.num_nodes, 
                    feature_types=[self.guidance_tar], return_dict=True
                )[self.guidance_tar]
                labels = torch.nan_to_num(labels).to(h.device)

                logits = self.guidance_head(h)
                loss = (logits - labels).pow(2)

        elif self.guidance_mod == 'edge':
            edge_index = batched_graph.edge_index
            if self.guidance_tar == 'feat':
                if edge_index.shape[1] > 0:
                    labels = batched_graph.edge_feat  # TODO: time-dependent edge_attr sampling
                    labels = torch.nan_to_num(labels).to(h.device)

                    src, dst = edge_index
                    logits = self.guidance_head(h, src, dst)
                    loss = (logits - labels).pow(2)
                else:
                    loss = torch.FloatTensor([0]).to(h.device)
                    loss.requires_grad_()

            else:
                if edge_index.shape[1] > 0:
                    labels = get_features(
                        edge_index.cpu(), num_nodes=batched_graph.num_nodes, 
                        feature_types=[self.guidance_tar], return_dict=True
                    )[self.guidance_tar]
                    labels = torch.nan_to_num(labels).to(h.device)

                    src, dst = edge_index
                    logits = self.guidance_head(h, src, dst)
                    loss = (logits - labels).pow(2)
                else:
                    loss = torch.FloatTensor([0]).to(h.device)
                    loss.requires_grad_()

        elif self.guidance_mod == 'graph':
            if self.guidance_tar == 'class':
                batch, labels = batched_graph.batch, batched_graph.labels
                labels = self.get_corrupted_labels(labels, t)

                logits = self.guidance_head(h, batch)
                loss = F.cross_entropy(logits, labels)
            
            elif self.guidance_tar == 'binary':
                batch, labels = batched_graph.batch, batched_graph.y

                logits = self.guidance_head(h, batch)
                is_labeled = labels == labels
                loss = F.binary_cross_entropy_with_logits(logits[is_labeled], labels.float()[is_labeled])

            elif self.guidance_tar == 'regression':
                batch, labels = batched_graph.batch, batched_graph.y

                logits = self.guidance_head(h, batch)
                is_labeled = labels == labels
                loss = (logits[is_labeled] - labels[is_labeled]).pow(2)

            else:
                batch, labels = batched_graph.batch, batched_graph.prop_attr

                logits = self.guidance_head(h, batch)
                loss = (logits - labels).pow(2)

        else:
            raise ValueError()

        return loss

    def forward(self, batched_graph, **kwargs):
        # batched_graph.orig_edge_index = batched_graph.edge_index.clone()
        b = batched_graph.num_graphs
        device = batched_graph.edge_index.device
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        t, pt = self._sample_time(b, device, self.sample_time_method)
        t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

        # get noisy graph
        if self.parametrization == 'x0':
            self._q_sample_and_set_xt_given_x0(batched_graph, t_edge)
        elif self.parametrization == 'xt':
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_edge)
        else:
            raise ValueError()

        batched_graph = self.setup_model_input(batched_graph)
        h = self.model.get_latent(batched_graph, t_node)
        loss = self.guidance_loss(h, batched_graph, t=t, **kwargs)

        # loss_prev = self.Lt_history.gather(dim=0, index=t)
        # new_Lt_history = (0.1 * loss + 0.9 * loss_prev).detach()
        # self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        # self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(loss))

        return loss.mean()

    def guidance_step(self, batched_graph, t_node, step_size=0.1, stability_coef=1e-2, num_steps=5, 
                      mask=None, **kwargs):
        if mask is None:
            mask = 1.

        self.model.eval()
        self.guidance_head.eval()
        edge_logits, h_list = self.model(batched_graph, t_node)

        src, dst = batched_graph.full_edge_index[0],  batched_graph.full_edge_index[1]
        h = h_list[-1]

        kl_loss = torch.nn.KLDivLoss(log_target=True)
        delta = torch.nn.Parameter(torch.zeros_like(h), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=step_size)

        h, edge_logits = h.detach(), edge_logits.detach()
        with torch.enable_grad():
            for _ in range(num_steps):
                h_current = h + mask * delta
                target_loss = self.guidance_score(h_current, batched_graph, **kwargs).sum()
                new_logits = self.model.forward_decoder(h_current, src, dst)

                kl = kl_loss(new_logits, edge_logits)
                loss = -target_loss + stability_coef * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        edge_logits = self.model.forward_decoder(h_current, src, dst)
        return edge_logits

    # overwrite the p_sample function
    def p_sample(self, batched_graph, t_node, t_edge, full_edge_attr=None, guidance_flag=False, step_size=0.1, 
                 stability_coef=1e-2, num_steps=5, mask=None, thres=None, **kwargs):
        batched_graph = self.setup_model_input(batched_graph)
        if guidance_flag:
            edge_logits = self.guidance_step(
                batched_graph, t_node, step_size=step_size, stability_coef=stability_coef, 
                num_steps=num_steps, mask=mask, **kwargs
            )
        else:
            edge_logits = self.model(batched_graph, t_node)[0]

        assert edge_logits.size(1) == self.num_edge_classes
        log_full_edge_recon = F.log_softmax(edge_logits, dim=1)

        if self.parametrization == 'x0':
            if full_edge_attr is not None:  # inpaint
                is_edge_indices = full_edge_attr.nonzero(as_tuple=True)[0]
                log_full_edge_attr = index_to_log_onehot(full_edge_attr, self.num_edge_classes)
                log_full_edge_recon[is_edge_indices] = log_full_edge_attr[is_edge_indices]

            log_model_pred_edge = self._q_posterior(
                log_x_start=log_full_edge_recon, log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, 
                log_final_prob=self.log_final_prob_edge
            )
        elif self.parametrization == 'xt':
            log_model_pred_edge = log_full_edge_recon

        log_out_edge = self.log_sample_categorical(log_model_pred_edge, self.num_edge_classes, thres=thres)

        return log_out_edge
    
    # overwrite the sample function
    def sample(self, batched_graph=None, num_samples=10, inpaint_flag=False, inpaint_every_step=True, 
               device='cuda:0', guidance_flag=False, step_size=0.1, stability_coef=1e-2, num_steps=5, 
               mask=None, thres=None, num_sample_steps=None, **kwargs):
        self.freeze_for_sample()
        self.model.eval()

        sampling_steps = np.arange(self.num_timesteps)
        num_sample_steps = self.num_timesteps if num_sample_steps is None else num_sample_steps
        num_sample_steps = int(num_sample_steps) 
        if num_sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, num_sample_steps)
            idx = np.array(idx * (self.num_timesteps-1), int)
            sampling_steps = sampling_steps[idx]

        full_edge_attr = None
        if batched_graph is None:
            batched_graph = self.initial_graph_sampler.sample(num_samples)
            batched_graph.to(device)
            batched_graph = self._prepare_data_for_sampling(batched_graph)
            num_edges = batched_graph.edges_per_graph.sum()
        else:
            batched_graph.to(device)
            try:
                num_edges = batched_graph.edges_per_graph.sum()
            except:
                num_edges = batched_graph.edges_per_graph
            t_edge = torch.full((num_edges,), self.num_timesteps - 1, device=device, dtype=torch.long)
            self._q_sample_and_set_xt_given_x0(batched_graph, t_edge)
            if inpaint_every_step:
                full_edge_attr = batched_graph.full_edge_attr
 
        try:
            num_nodes = batched_graph.nodes_per_graph.sum()
        except:
            num_nodes = batched_graph.nodes_per_graph

        # for t in tqdm.tqdm(reversed(range(0, self.num_timesteps))):
        for t in reversed(sampling_steps):
            # print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=device, dtype=torch.long)

            log_full_edge_attr_tmin1 = self.p_sample(
                batched_graph, t_node, t_edge, full_edge_attr, guidance_flag, step_size=step_size,
                stability_coef=stability_coef, num_steps=num_steps, mask=mask, thres=thres, **kwargs
            )
            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            torch.cuda.empty_cache()

        full_edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        if inpaint_flag:
            inpaint_indices = batched_graph.full_edge_attr.nonzero(as_tuple=True)[0]
            full_edge_attr[inpaint_indices] = batched_graph.full_edge_attr[inpaint_indices]

        is_edge_indices = full_edge_attr.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)
        batched_graph.edge_index = edge_index 

        edge_attr = full_edge_attr[is_edge_indices]
        batched_graph.edge_attr = edge_attr

        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']

        return batched_graph