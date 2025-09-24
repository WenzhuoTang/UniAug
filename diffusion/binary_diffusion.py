# modified from https://github.com/ZeWang95/BinaryLatentDiffusion/blob/main/models/binarylatent.py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    to_dense_batch, to_dense_adj, dense_to_sparse, 
    to_undirected, add_self_loops, add_remaining_self_loops
)

from diffusion.noise_schedule import BernoulliNoiseScheduler


class BernoulliOneHotDiffusion(torch.nn.Module):
    def __init__(self, model, num_timesteps=64, beta_type='linear_cumprod', loss_type='mean', 
                 use_softmax=False, p_flip=True, lbd=0.1, pi=0., parametrization='x0', **kwargs):
        super().__init__()
        self.lbd = lbd
        self.p_flip = p_flip
        self.loss_type = loss_type
        self.model = model
        self.use_softmax = use_softmax
        self.parametrization = parametrization

        self.pi = pi
        self.num_timesteps = num_timesteps
        self.scheduler = BernoulliNoiseScheduler(self.num_timesteps, beta_type=beta_type, pi=self.pi)

    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        return t

    def q_sample(self, x_0, t):
        x_t = self.scheduler(x_0, t) # 1<= t <=T
        return x_t

    def prob2onehot(self, prob):
        return torch.cat([(1 - prob).unsqueeze(-1), prob.unsqueeze(-1)], dim=-1)

    def _train_loss(self, batched_data):
        """ 
        kl_loss: simplified objective
        aux_loss: auxiliary VLB objective
        """
        full_edge_0, batch = batched_data.full_edge_attr.float(), batched_data.batch
        b, device = len(batch.unique()), batch.device

        # choose what time steps to compute loss at
        t = self.sample_time(b, device)
        t_edge = t.repeat_interleave(batched_data.num_full_edges)

        # make adj noisy
        full_edge_t_prob = self.q_sample(full_edge_0, t_edge)
        full_edge_t = torch.bernoulli(full_edge_t_prob).float()
        is_edge_indices = full_edge_t.nonzero(as_tuple=True)[0]
        edge_index = batched_data.full_edge_index[:, is_edge_indices]
        edge_index = add_self_loops(torch.cat([edge_index, edge_index.flip(0)],dim=-1))[0]
        batched_data.edge_index = edge_index

        # denoise
        batched_t = t[batch]
        full_edge_0_hat_logits = self.model(batched_data, batched_t - 1)[0]

        if self.parametrization == 'x0_flipped':
            full_edge_0_hat_logits = (1 - self.prob2onehot(full_edge_t)) * full_edge_0_hat_logits
            kl_loss = F.binary_cross_entropy_with_logits(
                full_edge_0_hat_logits, self.prob2onehot(full_edge_0), reduction='none'
            )
        elif self.parametrization == 'x0':
            kl_loss = F.binary_cross_entropy_with_logits(
                full_edge_0_hat_logits, self.prob2onehot(full_edge_0), reduction='none'
            )
        elif self.parametrization == 'xt':
            pass

        if torch.isinf(kl_loss).max():
            import ipdb; ipdb.set_trace()

        if self.loss_type == 'weighted':
            weight = (1 - ((t-1) / self.num_timesteps))
        elif self.loss_type == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        with torch.no_grad():
            acc = (((full_edge_0_hat_logits[..., 1] > full_edge_0_hat_logits[..., 0]).float() \
                == full_edge_0.view(-1)).float()).sum() / float(full_edge_0.numel())

        if self.lbd > 0:
            # ftr = (((t-1)==0)*1.0).view(-1, 1, 1)
            ftr = ((t_edge-1)==0).float().unsqueeze(-1)

            full_edge_0_logits = F.softmax(full_edge_0_hat_logits, -1)
            full_edge_t_logits = self.prob2onehot(full_edge_t_prob)

            p_EV_qxtmin_x0 = self.scheduler(full_edge_0_logits, t_edge-1)

            q_one_step = self.scheduler.one_step(full_edge_t_logits, t_edge)
            probs = p_EV_qxtmin_x0 * q_one_step
            probs = probs / (probs.sum(-1, keepdims=True) + 1e-6)

            full_edge_tm1_logits = probs * (1 - ftr) + full_edge_0_logits * ftr
            p_EV_qxtmin_x0_gt = self.scheduler(self.prob2onehot(full_edge_0), t_edge-1)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            full_edge_tm1_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)

            if torch.isinf(full_edge_tm1_logits).max() or torch.isnan(full_edge_tm1_logits).max():
                import ipdb; ipdb.set_trace()

            aux_loss = F.binary_cross_entropy(
                full_edge_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), 
                full_edge_tm1_gt.clamp(min=0.0, max=1.0), reduction='none'
            )

            aux_loss = (weight * aux_loss).mean()
            loss = self.lbd * aux_loss + loss

        stats = {'loss': loss, 'bce_loss': kl_loss, 'acc': acc}

        if self.lbd > 0:
            stats['aux loss'] = aux_loss
        return stats

    # XXX: Modified from https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py
    # XXX: In the original paper, guidance_model = self.model + RegressionHead
    # XXX: The RegressionHead is trained on noisy input and noisy labels
    def guidance_step(self, full_edge_index, h, batched_t, guidance_layer='last', step_size=0.1, 
                      stability_coef=1e-2, num_steps=5, mask=None):
        if mask is None:
            mask = 1.
        
        kl_loss = torch.nn.KLDivLoss(log_target=True)
        delta = torch.nn.Parameter(torch.zeros_like(h), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=step_size)

        row, col = full_edge_index[0], full_edge_index[1]
        edge_logits = self.model.final_out(h, row, col)

        with torch.enable_grad():
            for _ in range(num_steps):
                h_current = h + mask * delta

                if guidance_layer == "last":
                    target_loss = self.model.guidance_score(
                        None, batched_t-1, h=h_current
                    ).sum()
                    new_logits = self.model.final_out(h_current, row, col)
                elif guidance_layer == "first":
                    # out = self.network.forward(
                    #     None, t, attn_mask, token_embed=h_current
                    # )
                    # target_loss = self.network.guidance_score(
                    #     None, t, attn_mask, sequence_output=out['sequence_output']
                    # ).sum()
                    # new_logits = out['logits']
                    pass

                kl = kl_loss(new_logits, edge_logits)
                loss = -target_loss + stability_coef * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        edge_logits = self.model.final_out(h_current, row, col)
        return edge_logits

    @torch.no_grad()
    def sample_step(self, full_edge_t, t, i, sampling_steps, batched_data=None, temp=1., guidance=False, guidance_layer='last', 
                    step_size=0.1, stability_coef=1e-2, num_steps=5, node_mask=None, device='cuda'):
        batched_data = batched_data.detach().clone()
        batch, num_full_edges = batched_data.batch, batched_data.num_full_edges
        b = len(batch.unique())
        t = t * torch.ones(b, device=device, dtype=torch.long)
        batched_t = t[batch]

        is_edge_indices = full_edge_t.nonzero(as_tuple=True)[0]
        edge_index = batched_data.full_edge_index[:, is_edge_indices]
        edge_index = add_self_loops(torch.cat([edge_index, edge_index.flip(0)],dim=-1))[0]
        batched_data.edge_index = edge_index.detach()

        full_edge_0_logits, h_list = self.model(batched_data, batched_t - 1)
        if guidance:
            full_edge_0_logits = self.guidance_step(
                batched_data.full_edge_index, h_list[-1], batched_t, guidance_layer=guidance_layer, 
                step_size=step_size, stability_coef=stability_coef, num_steps=num_steps, mask=node_mask
            )
        # scale by temperature
        full_edge_0_logits = full_edge_0_logits / temp

        full_edge_0_prob = F.softmax(full_edge_0_logits, -1)
        if self.p_flip:
            full_edge_0_prob = self.prob2onehot(
                (1 - full_edge_t) * full_edge_0_prob[..., 1]
                + full_edge_t * (1 - full_edge_0_prob[..., 1])
            )

        if not t[0].item() == 1:
            t_p = sampling_steps[i + 1] * torch.ones(b, device=device, dtype=torch.long)
            edge_tp = t_p.repeat_interleave(num_full_edges)
            p_EV_qxtmin_x0 = self.scheduler(full_edge_0_prob, edge_tp)

            # q_one_step = self.prob2onehot(full_edge_t)
            # for mns in range(sampling_steps[i] - sampling_steps[i+1]):
            #     q_one_step = self.scheduler.one_step(q_one_step, edge_tp - mns)

            # probs = p_EV_qxtmin_x0 * q_one_step
            probs = p_EV_qxtmin_x0
            probs = probs / probs.sum(-1, keepdims=True)
            probs = probs[..., 1]
            full_edge_new = torch.bernoulli(probs)

        else:
            full_edge_new = full_edge_0_prob.argmax(-1)

        return full_edge_new
    
    # XXX: impainting: large t_start, noisy_input=False
    @torch.no_grad()
    def sample(self, batched_data=None, t_start=1, noisy_input=True, t_sample=0, impaint_flag=False, 
               temp=1.0, num_sample_steps=None, b=5, num_node=None, return_all=False, edge_mask=None, 
               guidance=False, guidance_layer='last', step_size=0.1, stability_coef=1e-2, 
               num_steps=5, node_mask=None, device='cuda', **kwargs):
        assert 1 <= t_start <= self.num_timesteps
        sampling_steps = np.arange(t_start, self.num_timesteps + 1)
        if num_sample_steps is not None and num_sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, num_sample_steps)
            idx = np.array(idx * (self.num_timesteps - 1), int)
            sampling_steps = sampling_steps[idx]
        else:
            num_sample_steps = self.num_timesteps

        if batched_data is not None:
            full_edge_0, batch = batched_data.full_edge_attr.float(), batched_data.batch
            num_full_edges = batched_data.num_full_edges
            b, device = len(batch.unique()), batch.device

            if impaint_flag:
                full_edge_0_indices = full_edge_t.nonzero(as_tuple=True)[0]
            
            if noisy_input:
                assert 0 <= t_sample < num_sample_steps
                # make adj noisy
                t_sample = sampling_steps[-(1 + t_sample)] * torch.ones(b, device=device, dtype=torch.long)
                t_edge = t_sample.repeat_interleave(num_full_edges)
                full_edge_t = torch.bernoulli(self.q_sample(full_edge_0, t_edge)).float()
            else:
                full_edge_t = full_edge_0.detach().clone()
        else:
            num_full_nodes = num_node * torch.ones(b, device=device, dtype=int)
            num_full_edges = num_node * (num_node - 1) // 2 * torch.ones(b, device=device, dtype=int)
            batch = torch.arange(b, device=device, dtype=torch.long).repeat_interleave(num_full_nodes)
            full_edge_t = torch.zeros(b, device=device, dtype=torch.float32).repeat_interleave(num_full_edges)

        if edge_mask is not None:
            full_edge_t = full_edge_t * (1 - edge_mask)

        if return_all:
            full_edge_all = [full_edge_t]

        self.model.eval()
        sampling_steps = sampling_steps[::-1]
        for i, t in enumerate(sampling_steps):
            full_edge_t = self.sample_step(
                full_edge_t, t, i, sampling_steps, batched_data=batched_data, temp=temp, guidance=guidance, 
                guidance_layer=guidance_layer, step_size=step_size, stability_coef=stability_coef, num_steps=num_steps, 
                node_mask=node_mask, device=device
            )

            if impaint_flag:
                full_edge_t[full_edge_0_indices] = 1.
            if edge_mask is not None:
                full_edge_t = full_edge_t * (1 - edge_mask)
            if return_all:
                full_edge_all.append(full_edge_t)

        if return_all:
            return torch.cat(full_edge_all, 0)
        else:
            return full_edge_t
    
    def forward(self, batched_data):
        return self._train_loss(batched_data)['loss']

