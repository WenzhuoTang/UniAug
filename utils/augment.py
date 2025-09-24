import time
import torch
from tqdm import trange
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .loader import load_sde
from .infonce import InfoNCE
from .solver import (
    LangevinCorrector, 
    ReverseDiffusionPredictor, 
    mask_x, mask_adjs, gen_noise, get_score_fn
)
from utils.graph_utils import (
    convert_dense_to_rawpyg, 
    convert_sparse_to_dense, 
    extract_graph_feature,
    convert_dense_adj_to_sparse_with_attr,
)
from dataset.subgraph import (
    graph_to_segments,
    segments_to_graph
)
from dataset.loader import MultiEpochsPYGDataLoader

__all__ = ['build_augmentation_dataset']

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')

# -------- get negative samples for infoNCE loss --------
def get_negative_indices(y_true, n_sample=10):
    if torch.isnan(y_true).sum() != 0:
        print('y_true', (y_true==y_true).size(), (y_true==y_true).sum())
        return None
    y_true = torch.nan_to_num(y_true, nan=0.)
    task_num = y_true.size(1)
    diffs = torch.abs(y_true.view(-1,1,task_num) - y_true.view(1,-1,task_num)).mean(dim=-1)
    diffs_desc_indices = torch.argsort(diffs, dim=1, descending=True)
    return diffs_desc_indices[:, :n_sample]


def inner_sampling(generator, x, adj, sde_x, sde_adj, diff_steps, flags=None, perturb_ratio=0.1, 
                   snr=0.5, scale_eps=0., n_steps=1, out_steps=1, device='cuda'):
    score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=perturb_ratio)
    score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=perturb_ratio)
    predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=perturb_ratio)
    corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
    predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=perturb_ratio)
    corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
    x, adj = mask_x(x, flags), mask_adjs(adj, flags)
    total_sample_steps = out_steps
    timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=device)[-diff_steps:]
    with torch.no_grad():
        # -------- Reverse diffusion process --------
        for i in range(diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(adj.shape[0], device=t.device) * t
            _x = x
            x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)
            _x = x
            x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
    return x_mean.detach(), adj_mean.detach()

## -------- Main function for augmentation--------##
def build_augmentation_dataset(model, generator, labeled_loader, use_sudo_label=False, device='cuda', 
                               sde_x_config={}, sde_adj_config={}, feat_dim=68, topk=5, strategy='add_once', 
                               n_steps=1, n_negative=1, out_steps=1, perturb_ratio=0.1, snr=0.5, scale_eps=0., 
                               n_jobs=1, pooling=None, aug_x=True, aug_adj=True, feat_to_compare='extract',
                               augment_all_graphs=False, **kwargs):
    infonce_paired = InfoNCE(temperature=0.05)
    criterion = cls_criterion
    prob_func = torch.nn.Sigmoid()

    kept_pyg_list = []
    augmented_pyg_list = []
    augment_fails = 0

    for step, batch_data in enumerate(labeled_loader):
        model.eval()
        batch_data_list = batch_data.to_data_list()
        batch_data = batch_data.to(device)
        if batch_data.x.shape[0] > 1:
            with torch.no_grad():
                y_pred_logits = model(batch_data)[0]
            y_true_all, batch_index = batch_data.y.to(torch.float32), batch_data.batch
            is_labeled = y_true_all == y_true_all
            y_pred_logits[~is_labeled], y_true_all[~is_labeled] = 0, 0

            selected_topk = topk
            if augment_all_graphs or topk > y_true_all.size(0):
                selected_topk = y_true_all.size(0)

            topk_indices = torch.topk(
                criterion(y_pred_logits.view(y_true_all.size()).to(torch.float32), y_true_all).view(y_true_all.size()).sum(dim=-1),
                selected_topk, largest=False, sorted=True
            ).indices
            augment_mask = torch.zeros(y_pred_logits.size(0)).to(y_pred_logits.device).scatter_(0, topk_indices, 1).bool()

            if use_sudo_label:
                augment_labels = torch.softmax(y_pred_logits[augment_mask], dim=-1)
            else: 
                augment_labels = y_true_all[augment_mask]

            if strategy.split('_')[0] == 'add':
                kept_indices = list(range(y_pred_logits.size(0)))
            elif strategy.split('_')[0] == 'replace':
                kept_indices = (~augment_mask).nonzero().view(-1).cpu().tolist()
            else:
                raise NotImplementedError(f"not implemented strategy {strategy}.")
            for kept_index in kept_indices:
                kept_pyg_list.append(batch_data_list[kept_index])

            batch_dense_x, batch_dense_adj, batch_node_mask = \
                convert_sparse_to_dense(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr, augment_mask=None)
            ori_x, ori_adj = batch_dense_x.clone().to(torch.float32), batch_dense_adj.clone().to(torch.float32)

            # TODO: This is a temporary fix. Need to provide a clear interface. only augment the struture features
            ## extract graph feature
            ori_stru_feat = ori_x[..., -feat_dim:]
            if feat_to_compare == 'structure':
                ori_feats = ori_stru_feat.sum(1)
            elif feat_to_compare == 'extract':
                ori_feats = extract_graph_feature(ori_stru_feat, ori_adj, node_mask=batch_node_mask, pooling=pooling)
            else:
                raise NotImplementedError(f'Unsupported feat_to_compare {feat_to_compare}')

            neg_indices = get_negative_indices(y_true_all, n_sample=n_negative) # B x n_sample
            neg_indices = neg_indices[augment_mask]
        
            # sde
            sde_x_config['num_scales'] = sde_adj_config['num_scales'] = total_sample_steps = out_steps
            sde_x = load_sde(sde_x_config)
            sde_adj = load_sde(sde_adj_config)

            batch_dense_x, batch_dense_adj = batch_dense_x[augment_mask], batch_dense_adj[augment_mask]
            
            # perturb x 
            x_to_aug = batch_dense_x[..., -feat_dim:]
            x_from_data = batch_dense_x[..., :-feat_dim]
            peturb_t = torch.ones(batch_dense_adj.shape[0]).to(device) * (sde_adj.T - 1e-3) + 1e-3
            mean_x, std_x = sde_x.marginal_prob(x_to_aug, peturb_t)
            z_x = gen_noise(x_to_aug, flags=batch_node_mask[augment_mask], sym=False, perturb_ratio=perturb_ratio)
            perturbed_x = mean_x + std_x[:, None, None] * z_x
            perturbed_x = mask_x(perturbed_x, batch_node_mask[augment_mask])
            
            # perturb adj
            mean_adj, std_adj = sde_adj.marginal_prob(batch_dense_adj, peturb_t)
            z_adj = gen_noise(batch_dense_adj, flags=batch_node_mask[augment_mask], sym=True, perturb_ratio=perturb_ratio)
            perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
            perturbed_adj = mask_adjs(perturbed_adj, batch_node_mask[augment_mask])

            timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=device)[-out_steps:]
            def get_aug_grads(prediction_model, inner_output_data):
                prediction_model.eval()
                inner_output_x, inner_output_adj = inner_output_data
                inner_output_adj = mask_adjs(inner_output_adj, batch_node_mask[augment_mask])
                inner_output_x = mask_x(inner_output_x, batch_node_mask[augment_mask])

                inner_output_x,  inner_output_adj = inner_output_x.requires_grad_(), inner_output_adj.requires_grad_()
                with torch.enable_grad():
                    inner_x_all, inner_adj_all = inner_output_x, inner_output_adj
                    edge_index, edge_attr = convert_dense_adj_to_sparse_with_attr(inner_adj_all, batch_node_mask[augment_mask])
                    bdata_batch_index = batch_node_mask[augment_mask].nonzero()[:,0]
                    bdata_x = torch.cat([x_from_data, inner_x_all], dim=-1)[batch_node_mask[augment_mask]]
                    bdata_y = augment_labels.view(inner_x_all.size(0), -1)
                    # node_feature_encoded = estimate_feature_embs(inner_x_all[batch_node_mask[augment_mask]], batch_data, prediction_model, obj='node')
                    # edge_attr_encoded = estimate_feature_embs(edge_attr, batch_data, prediction_model, obj='edge')
                    # bdata = Data(x=node_feature_encoded, edge_index=edge_index, edge_attr=edge_attr_encoded, y=bdata_y, batch=bdata_batch_index)

                    bdata = Data(x=bdata_x, 
                                 edge_index=edge_index, 
                                 edge_attr=edge_attr, 
                                 y=bdata_y,
                                 batch=bdata_batch_index)
                    
                    if inner_output_x.shape[0] > 1:
                        preds = prediction_model(bdata)[0]
                        is_labeled = bdata.y == bdata.y
                        bdata_target = bdata.y.to(torch.float32)[is_labeled]
                        bdata_probs = prob_func(preds.view(bdata.y.size()).to(torch.float32)[is_labeled])
                        loss_y = torch.log((bdata_probs * bdata_target + (1 - bdata_probs) * (1 - bdata_target)).clamp_min(1e-18)).mean() # maximize log likelihood
                        
                        if feat_to_compare == 'sum':
                            aug_feats = inner_x_all.sum(1)
                        elif feat_to_compare == 'extract':
                            aug_feats = extract_graph_feature(inner_x_all, inner_adj_all, node_mask=batch_node_mask[augment_mask], pooling=pooling)
                        else:
                            raise NotImplementedError(f'Unsupported feat_to_compare {feat_to_compare}')

                        query_structure_embed, pos_structure_embed, neg_structure_embed = aug_feats, ori_feats[augment_mask], ori_feats[neg_indices]
                        loss_structure = infonce_paired(query_structure_embed, pos_structure_embed, neg_structure_embed) # maximize infonce == minimize mutual information                        

                        total_loss = loss_y + loss_structure
                        aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [inner_output_x, inner_output_adj])
                    else:
                        aug_grad_x, aug_grad_adj = None, None
                return aug_grad_x, aug_grad_adj

            score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=perturb_ratio)
            score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=perturb_ratio)
            predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=perturb_ratio)
            corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
            predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=perturb_ratio)
            corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
            outer_iters = trange(0, (out_steps), desc = '[Outer Sampling]', position = 1, leave=False)
            for i in outer_iters:
                inner_output_x, inner_output_adj = inner_sampling(generator, perturbed_x, perturbed_adj, sde_x, sde_adj, out_steps-i, batch_node_mask[augment_mask],
                                                                  perturb_ratio, snr, scale_eps, n_steps, out_steps, device)
                inner_output_x = inner_output_x if aug_x else x_to_aug
                inner_output_adj = inner_output_adj if aug_adj else batch_dense_adj
                aug_grad_x, aug_grad_adj = get_aug_grads(model, [inner_output_x, inner_output_adj])
                with torch.no_grad():
                    t = timesteps[i]
                    vec_t = torch.ones(perturbed_adj.shape[0], device=t.device) * t
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = corrector_obj_x.update_fn(perturbed_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = corrector_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_adj)
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = predictor_obj_x.update_fn(perturbed_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = predictor_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_adj)

            perturbed_adj_mean = mask_adjs(perturbed_adj_mean, batch_node_mask[augment_mask])
            perturbed_x_mean = mask_x(perturbed_x_mean, batch_node_mask[augment_mask])
            augmented_x, augmented_adj = perturbed_x_mean.cpu(), perturbed_adj_mean.cpu()

            new_x = augmented_x if aug_x else x_to_aug.detach().cpu()
            if ori_x.shape[-1] > feat_dim:
                new_x = torch.cat([x_from_data.detach().cpu(), augmented_x], dim=-1)
            new_adj = augmented_adj if aug_adj else batch_dense_adj.detach().cpu()
            batch_augment_pyg_list = convert_dense_to_rawpyg(new_x, new_adj, y_true_all[augment_mask], n_jobs=n_jobs)

            augment_indices = augment_mask.nonzero().view(-1).cpu().tolist()
            augmented_pyg_list_temp = []
            for pyg_data in batch_augment_pyg_list:
                if not isinstance(pyg_data, int):
                    augmented_pyg_list_temp.append(pyg_data)
                elif strategy.split('_')[0] == 'add':
                    pass
                else:
                    augment_fails += 1
                    kept_pyg_list.append(batch_data_list[augment_indices[pyg_data]])

            augmented_pyg_list.extend(augmented_pyg_list_temp)

    new_dataset = NewDataset(kept_pyg_list, num_fail=augment_fails)
    return new_dataset


def get_positive_indices(y_true, n_sample=10):
    if torch.isnan(y_true).sum() != 0:
        print('y_true', (y_true==y_true).size(), (y_true==y_true).sum())
        return None
    y_true = torch.nan_to_num(y_true, nan=0.)
    if len(y_true.size()) < 2:
        y_true = torch.nn.functional.one_hot(y_true).to(torch.float32)
    task_num = y_true.size(1)
    diffs = torch.abs(y_true.view(-1,1,task_num) - y_true.view(1,-1,task_num)).mean(dim=-1)
    diffs_desc_indices = torch.argsort(diffs, dim=1, descending=True)
    return diffs_desc_indices[:, -n_sample:]


def augment_segments(model, generator, data, criterion, aug_orig=False, device='cuda', sde_x_config={}, sde_adj_config={}, topk=None, 
                     n_steps=1, out_steps=1, perturb_ratio=0.1, snr=0.5, scale_eps=0., aug_x=True, aug_adj=True, 
                     max_node_num=200, batch_size=32, cutoff=0.5, infonce=False, pooling=None, n_negative=1, **kwargs):
    if data.part_list is None:
        raise NotImplementedError(f'Please run extract_segments')
    prob_func = torch.nn.Sigmoid()

    model.eval()
    aug_data = data.detach().clone()
    if not aug_orig:
        with torch.no_grad():
            try:
                _, x_hidden = model(data)
            except:
                x_hidden = data.x
        aug_data.x = x_hidden

    if infonce:
        y_true, train_mask= data.y, data.train_mask
        infonce_paired = InfoNCE(temperature=0.05)
        neg_indices = get_negative_indices(y_true, n_sample=n_negative)[train_mask]
        pos_indices = get_positive_indices(y_true, n_sample=n_negative)[train_mask]

    def get_aug_grads(prediction_model, data):
        prediction_model.eval()
        x, y, train_mask = data.x, data.y, data.train_mask
        adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[0]
        if infonce:
            ori_feats = extract_graph_feature(x, adj, pooling=pooling)
        x, adj = x.requires_grad_(), adj.requires_grad_()
        with torch.enable_grad():
            edge_index, edge_attr = convert_dense_adj_to_sparse_with_attr(adj)
            temp_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            preds = prediction_model(temp_data, input_hidden=not aug_orig)[0]
            loss_y = criterion(preds[train_mask], y[train_mask]).mean()
            # bdata_probs = prob_func(preds.view(bdata.y.size()).to(torch.float32)[is_labeled])
            # loss_y = torch.log((bdata_probs * bdata_target + (1 - bdata_probs) * (1 - bdata_target)).clamp_min(1e-18)).mean() # maximize log likelihood
                
            if infonce:
                aug_feats = extract_graph_feature(x, adj, pooling=pooling)
                query_structure_embed, pos_structure_embed, neg_structure_embed = aug_feats, ori_feats[pos_indices], ori_feats[neg_indices]
                loss_structure = infonce_paired(query_structure_embed, pos_structure_embed, neg_structure_embed) # maximize infonce == minimize mutual information
                total_loss = loss_y + loss_structure
            else:
                total_loss = loss_y
            aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [x, adj])

        return aug_grad_x, aug_grad_adj
    
    # calculate grad
    aug_grad_x, aug_grad_adj = get_aug_grads(model, aug_data)

    # sde
    sde_x_config['num_scales'] = sde_adj_config['num_scales'] = total_sample_steps = out_steps
    sde_x = load_sde(sde_x_config)
    sde_adj = load_sde(sde_adj_config)

    # graph to segments
    data_list, remaining_edge_index, remaining_edge_attr = graph_to_segments(
        aug_data, max_node_num=max_node_num, add_node_id=True, fill_zeros=True
    )
    new_data_list = []
    loader = MultiEpochsPYGDataLoader(data_list, batch_size=batch_size, shuffle=True)
    for step, batch_data in enumerate(loader):
        if topk is not None and step * batch_size > topk:
            new_data_list.extend(batch_data.to_data_list())
        else:
            batch_index = batch_data.batch
            batch_dense_x, batch_dense_adj, batch_node_mask = convert_sparse_to_dense(
                batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr, augment_mask=None
            )
            max_count = torch.unique(batch_index, return_counts=True)[1].max()
            batch_dense_y, _ = to_dense_batch(batch_data.y, batch=batch_index, max_num_nodes=max_count)
            batch_dense_node_id, _ = to_dense_batch(batch_data.node_id, batch=batch_index, max_num_nodes=max_count)

            # perturb x 
            peturb_t = torch.ones(batch_dense_adj.shape[0]).to(device) * (sde_adj.T - 1e-3) + 1e-3
            mean_x, std_x = sde_x.marginal_prob(batch_dense_x, peturb_t)
            z_x = gen_noise(batch_dense_x, flags=batch_node_mask, sym=False, perturb_ratio=perturb_ratio)
            perturbed_x = mean_x + std_x[:, None, None] * z_x
            perturbed_x = mask_x(perturbed_x, batch_node_mask)
            
            # perturb adj
            mean_adj, std_adj = sde_adj.marginal_prob(batch_dense_adj, peturb_t)
            z_adj = gen_noise(batch_dense_adj, flags=batch_node_mask, sym=True, perturb_ratio=perturb_ratio)
            perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
            perturbed_adj = mask_adjs(perturbed_adj, batch_node_mask)

            timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=device)[-out_steps:]
            score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=perturb_ratio)
            score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=perturb_ratio)
            predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=perturb_ratio)
            corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
            predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=perturb_ratio)
            corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps, perturb_ratio=perturb_ratio)
            outer_iters = trange(0, (out_steps), desc = '[Outer Sampling]', position=1, leave=False)

            # segment-wisely crop gradient
            aug_mask = batch_dense_node_id >= 0
            aug_node_id = batch_dense_node_id[aug_mask]
            aug_grad_x_batch = torch.zeros_like(batch_dense_x)
            aug_grad_adj_batch = torch.zeros_like(batch_dense_adj)
            aug_grad_x_batch[aug_mask] = aug_grad_x[aug_node_id]
            for b in range(len(aug_grad_adj_batch)):
                aug_grad_adj_batch[b, aug_mask[b], aug_mask[b]] = \
                    aug_grad_adj[batch_dense_node_id[b][aug_mask[b]], batch_dense_node_id[b][aug_mask[b]]]
            for i in outer_iters:
                with torch.no_grad():
                    t = timesteps[i]
                    vec_t = torch.ones(perturbed_adj.shape[0], device=t.device) * t
                    _x = perturbed_x.detach().clone()
                    perturbed_x, perturbed_x_mean = corrector_obj_x.update_fn(
                        perturbed_x, perturbed_adj, batch_node_mask, vec_t, aug_grad=aug_grad_x_batch
                    )
                    perturbed_adj, perturbed_adj_mean = corrector_obj_adj.update_fn(
                        _x, perturbed_adj, batch_node_mask, vec_t, aug_grad=aug_grad_adj_batch
                    )
                    _x = perturbed_x.detach().clone()
                    perturbed_x, perturbed_x_mean = predictor_obj_x.update_fn(
                        perturbed_x, perturbed_adj, batch_node_mask, vec_t, aug_grad=aug_grad_x_batch
                    )
                    perturbed_adj, perturbed_adj_mean = predictor_obj_adj.update_fn(
                        _x, perturbed_adj, batch_node_mask, vec_t, aug_grad=aug_grad_adj_batch
                    )
            perturbed_adj_mean = mask_adjs(perturbed_adj_mean, batch_node_mask)
            perturbed_x_mean = mask_x(perturbed_x_mean, batch_node_mask)
            augmented_x, augmented_adj = perturbed_x_mean.cpu(), perturbed_adj_mean.cpu()

            new_x = augmented_x if aug_x else batch_dense_x.detach().cpu()
            new_adj = augmented_adj if aug_adj else batch_dense_adj.detach().cpu()
            if cutoff is not None:
                new_adj = (new_adj > cutoff).to(torch.float32)
            batch_dense_inputs = {'y': batch_dense_y.cpu(), 'node_id': batch_dense_node_id.cpu()}
            new_data_list.extend(convert_dense_to_rawpyg(new_x, new_adj, **batch_dense_inputs))

    new_data = segments_to_graph(new_data_list, remaining_edge_index, remaining_edge_attr).to(device)
    return new_data


class NewDataset(InMemoryDataset):
    def __init__(self, data_list, num_fail=0, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_len = len(data_list)
        self.num_fail = num_fail
        # print('data_len', self.data_len, 'num_fail', num_fail)
        self.data, self.slices = self.collate(data_list)
    # def get_idx_split(self):
    #     return {'train': torch.arange(self.data_len, dtype = torch.long), 'valid': None, 'test': None}


if __name__ == '__main__':
    pass