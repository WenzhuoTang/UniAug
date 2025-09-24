import time
import wandb
import os.path as osp
import numpy as np
from typing import Iterable
from pprint import pformat
from tqdm import tqdm, trange
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, negative_sampling

from dataset.subgraph import graph_to_segments, segments_to_graph
from parsers import Parser, get_config
from utils import (
    dict_of_dicts_to_dict, set_seed
)
from utils.eval import evaluate_link_prediction
from utils.loader import (
    load_data, 
    load_sampler,
    load_predictor,
    load_downstream_model,
    load_optimizer_and_scheduler,
    load_diffusion_guidance_optim,
)
from dataset.loader import MultiEpochsPYGDataLoader
from dataset.misc import batched_to_list, get_diffusion_attributes
from dataset.property import get_properties, split_by_property
from diffusion.diffusion_utils import setdiff
from models.utils import PermIterator

torch.set_num_threads(4)


METRIC_DICT = {
    'cora': 'mrr',
    'citeseer': 'mrr',
    'pubmed': 'mrr',
    'ogbl-collab': 'hits@50',
    'ogbl-ddi': 'hits@20',
    'ogbl-ppa': 'hits@100',
    'ogbl-citation2': 'mrr',
}

### Training Loop
def train(model,
          predictor,
          data,
          train_pos,
          x,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):

    if alpha is not None:
        predictor.setalpha(alpha)

    model.train()
    predictor.train()
    if type(x) == torch.nn.Embedding:
        x = x.weight

    pos_train_edge = train_pos

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(
                tei,sparse_sizes=(data.num_nodes, data.num_nodes)
                ).to_device(pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj

        h = model(x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        if torch.isnan(loss).any():
            import ipdb; ipdb.set_trace()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

### Evaluation Loop
@torch.no_grad()
def test(data_name, model, predictor, data, x, batch_size=1024, use_val_edges=False):
    model.eval()
    predictor.eval()
    if type(x) == torch.nn.Embedding:
        x = x.weight

    pos_valid_edge = data['valid_pos']
    neg_valid_edge = data['valid_neg']
    pos_test_edge = data['test_pos']
    neg_test_edge = data['test_neg']

    adj = data.adj
    h = model(x, adj)

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm]).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm]).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_val_edges:
        h = model(x, data.full_adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm]).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm]).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    hits = False if data_name == 'ogbl-citation2' else True
    mrr = False if data_name in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'flickr', 'photo'] else True

    results = {}
    results['valid'] = evaluate_link_prediction(pos_valid_pred, neg_valid_pred, hits=hits, mrr=mrr)
    results['test'] = evaluate_link_prediction(pos_test_pred, neg_test_pred, hits=hits, mrr=mrr)

    return dict_of_dicts_to_dict(results)

### Main Entry Function 
def run(config, seed=10, augment=False, use_val_edges=False, remove_dup=True, train_guidance=False):
    set_seed(seed)

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data_params, model_params = config.data, config.model
    predictor_params, train_params = config.predictor, config.train

    data_name = data_params.data_name
    dataset = load_data(data_params, return_loader=False, cluster_path=cluster_path)
    data = dataset[0]

    edge_attr_keys = None
    if data_name == 'ogbl-collab':
        data.edge_feat = data.edge_year
        data.edge_feat = data.edge_feat - data.edge_feat.min()
        edge_attr_keys = ['edge_feat']

    if augment:
        # TODO: multiple samples
        # num_samples = augment_params.num_samples
        # for i in trange(1, 1 + num_samples, desc = '[Sample]', position = 1, leave=False):
        #     pass

        orig_data = data.clone()

        # use train_pos as edge_index
        data.edge_index = torch.cat((data.train_pos, data.train_pos[[1,0]]), dim=1)
        data = data.coalesce()
        if hasattr(data, 'stru_feat'):
            data.x = data.stru_feat
        
        augment_params = config.augment
        segment_flag = augment_params.segment_flag
        data = data.cpu()

        if segment_flag:
            data_list, remaining_edge_index, remaining_edge_attr = graph_to_segments(
                data, add_diff_attr=True, edge_attr_keys=edge_attr_keys
            )
        else:
            data = get_diffusion_attributes(data)
            data_list = [data]

        # get kmeans labels for self-guided diffusion
        kmeans, _, _, scaler_dict = torch.load(cluster_path)
        data_list, property_attr, _, _ = get_properties(data_list, scaler_dict=scaler_dict, nodes_and_edges=True)
        _, kmeans_labels = split_by_property(property_attr, kmeans=kmeans)
        for i in range(len(data_list)):
            data_list[i].kmeans_labels = torch.from_numpy(
                kmeans_labels[i].repeat(data_list[i].num_nodes)
            )

        ckpt_path = augment_params.ckpt_path
        if train_guidance:
            guidance_config = augment_params.guidance_config
            freeze_model = guidance_config.diffusion.freeze_model
            guidance_config, sampler, sampler_optim, sampler_sched = load_diffusion_guidance_optim(
                guidance_config, device, ckpt_path, freeze_model=freeze_model
            )

            ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
            guidance_ckpt_name = f"{ckpt_path.split('/')[-1].split('-')[0]}-guidance_{data_name}-r.{seed}-{ts}.pth"
            guidance_ckpt_dir = osp.join(*ckpt_path.split('/')[:-1])
            guidance_ckpt_path = osp.join(guidance_ckpt_dir, guidance_ckpt_name)

            num_epochs, batch_size = guidance_config.num_epochs, guidance_config.batch_size
            train_loader = MultiEpochsPYGDataLoader(data_list, batch_size=batch_size, shuffle=True)

            print('Training guidance model ...')
            guidance_params = {}
            
            for epoch in (pbar := trange(0, (num_epochs), desc = '[Epoch]', position = 1, leave=False)):
                losses = []
                for _, train_bdata in enumerate(train_loader):
                    train_bdata = train_bdata.to(device)
                    guidance_params['pos_edges'] = train_bdata.edge_index
                    loss = sampler(train_bdata, **guidance_params)
                    loss.backward()
                    losses.append(loss.item())

                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.0)
                    sampler_optim.step()

                tqdm_log = f"[EPOCH {epoch+1:04d}] | train loss: {np.mean(losses):.3e}"
                pbar.set_description(tqdm_log)

            save_flag = guidance_config.save_flag
            if save_flag:
                print(f'Saving to {guidance_ckpt_path}')
                torch.save({
                    'model_config': guidance_config,
                    'model_state_dict': sampler.state_dict(),
                }, guidance_ckpt_path)

        else:
            sampler = load_sampler(ckpt_path, device=device)

        batch_size = int(augment_params.batch_size)
        dataloader = MultiEpochsPYGDataLoader(data_list, batch_size=batch_size, shuffle=False)
        print(f'{len(dataloader)} batches in total.')

        print('Augmenting...')
        new_data_list = []
        sample_params = OmegaConf.to_container(augment_params.sample)

        for _, bdata in tqdm(enumerate(dataloader)):
            bdata = bdata.to(device)
            if hasattr(sampler, 'guidance_head'):
                sample_params['pos_edges'] = bdata.edge_index.clone()

            bdata = sampler.sample(
                bdata, device=device, **sample_params
            ).cpu()
            new_data_list.extend(batched_to_list(bdata))

        if segment_flag:
            data = segments_to_graph(new_data_list, remaining_edge_index, remaining_edge_attr)
        else:
            data = new_data_list[0]

        data.x = orig_data.x

        # recover edge_index from original data
        diff, _ = setdiff(data['edge_index'], orig_data['edge_index'], dim=1)
        _, additioanl_edges = setdiff(diff, orig_data['edge_index'], dim=1)
        if additioanl_edges.shape[1] > 0:
            data['edge_index'] = torch.cat([data['edge_index'], additioanl_edges], dim=1)

        # replace train_pos with upper triangle
        row, col = data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        data['train_pos'] = torch.stack([row, col])

        # # the edge_index is undirected, take the upper triangle
        # row, col = orig_data.edge_index
        # mask = row < col
        # row, col = row[mask], col[mask]
        # orig_data['train_pos'] = torch.stack([row, col])
        
        # # make sure new train_pos contrains original edges
        # diff, _ = setdiff(data['train_pos'], orig_data['train_pos'], dim=1)
        # _, additioanl_edges = setdiff(diff, orig_data['train_pos'], dim=1)
        # if additioanl_edges.shape[1] > 0:
        #     data['train_pos'] = torch.cat([data['train_pos'], additioanl_edges], dim=1)

        keys_to_copy = ['adj', 'train_val', 'valid_pos', 'valid_neg', 'test_pos', 'test_neg']
        for k in keys_to_copy:
            data[k] = orig_data[k]

        del data_list, new_data_list, sampler
        torch.cuda.empty_cache()
        print('Using augmented view to train')

    
    if remove_dup:
        data = data.coalesce()
    else:
        # recover duplicate edges
        row, col, value = data.adj.coalesce().coo()
        if value is not None:
            dup_idx = torch.where(value > 1)[0]
            if len(dup_idx) > 0:
                dup_edges = []
                for i in dup_idx:
                    dup_edges.append(
                        torch.stack([row[i], col[i]]).unsqueeze(-1).repeat(1, value[i] - 1)
                    )
                data.edge_index = torch.cat([data.edge_index, torch.cat(dup_edges, 1)], 1)

    data.adj = SparseTensor.from_edge_index(data.edge_index, None, [data.num_nodes, data.num_nodes])
    data = data.to(device)

    if use_val_edges:
        val_edge_index = to_undirected(data.valid_pos)
        eval_edge_index = torch.cat([data.edge_index, val_edge_index], 1)
        data.full_adj = SparseTensor.from_edge_index(eval_edge_index, None, [data.num_nodes, data.num_nodes])

    x, train_pos = data.x, data.train_pos
    if x is not None:
        emb_flag = False
        data.x = x.to(torch.float32)
        model_params.in_channels = x.size(1)
    else:
        emb_flag = True
        hidden_channels = model_params.in_channels = model_params.hidden_channels
        x = torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(x.weight)
    
    model_params.max_x = -1

    model = load_downstream_model(model_params).to(device)
    predictor = load_predictor(predictor_params).to(device)
    params_list = list(model.parameters()) + list(predictor.parameters())
    if emb_flag:
        params_list += list(x.parameters())

    optimizer, scheduler = load_optimizer_and_scheduler(params_list, train_params)

    epochs, batch_size = train_params.epochs, train_params.batch_size
    patience, eval_steps = train_params.patience, train_params.eval_steps
    maskinput = train_params.maskinput

    if hasattr(train_params, 'eval_start'):
        eval_start = train_params.eval_start
    else:
        eval_start = 0

    try:
        metric = METRIC_DICT[data_name]
    except:
        metric = 'hits@10'
    all_results = {}    
    valid_metric_list = []
    kill_cnt = 0

    try:
        increasealpha = train_params.increasealpha
    except:
        increasealpha = False

    for epoch in (pbar := trange(epochs, desc = '[Epoch]', position = 1, leave=False)):
        # loss = train(model, predictor, train_pos, x, optimizer, batch_size, emb_flag, neg_flag)
        alpha = max(0, min((epoch-5)*0.1, 1)) if increasealpha else None
        loss = train(model, predictor, data, train_pos, x, optimizer,
                     batch_size, maskinput, [], alpha)
        tqdm_log = f"[EPOCH {epoch + 1:04d}] | train loss: {loss:.3e}"

        if epoch >= eval_start and epoch % eval_steps == 0:
            results = test(data_name, model, predictor, data, x, batch_size, use_val_edges)
            wandb.log(results)

            for k in results.keys():
                if k not in all_results.keys():
                    all_results[k] = []
                all_results[k].append(results[k])

            # tqdm_log += f" | Train {metric}: {100 * results[f'train_{metric}']}"
            tqdm_log += f" | Valid {metric}: {100 * results[f'valid_{metric}']}"
            tqdm_log += f" | Test: {metric}: {100 * results[f'test_{metric}']}"

            valid_metric_list.append(results[f'valid_{metric}'])
            if max(valid_metric_list) <= valid_metric_list[-1] and max(valid_metric_list) < 1:
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > patience: 
                    print(f"Early stopped at epoch {epoch}")
                    break

            if kill_cnt == 0:
                tqdm.write(tqdm_log)

        pbar.set_description(tqdm_log)
    
    final_results = {}
    for k in all_results.keys():
        s, m = k.split('_')[0], k.split('_')[1]
        if s == 'test':
            try:
                selection = train_params.selection
            except:
                selection = 'best'

            if selection == 'best':
                best_idx = np.argmax(all_results[f'valid_{m}'])
            elif selection == 'last':
                best_idx = -1
            else:
                raise ValueError

            final_results[f'final_{m}'] = all_results[f'test_{m}'][best_idx]

    wandb.log(final_results)


if __name__ == "__main__":
    args, unknown = Parser().parse()
    cli = OmegaConf.from_dotlist(unknown)
    config = get_config(args.config, args.seed)
    config = OmegaConf.merge(config, cli)
    print(pformat(vars(config)))

    data_name = config.data.data_name
    augment = args.augment
    ckpt_path = config.augment.ckpt_path
    ckpt_prefix = ckpt_path.split('/')[-1].split('-r.')[0]

    if ckpt_prefix == 'full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide':
        cluster_path = 'data/misc/full_network_repository/processed/entr10_dens0.1_denm0_degm3_degv3_node4000_edge50000-feat_none-all_10clusters_ne.pt'
    elif ckpt_prefix == 'full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide':
        cluster_path = 'data/misc/full_network_repository/processed/entr10_dens0.1_denm50_degm3_degv3_node4000_edge50000-feat_none-ext_github_stargazers-all_10clusters_ne.pt'

    # thres = args.thres if args.thres is not None else config.augment.sample.thres
    ckpt_prefix = ckpt_prefix.split('_dnnm-')[0]
    thres = args.thres
    config.augment.sample.thres = thres  # overwrite

    group_postfix = f'aug{str(augment)}'
    try:
        selection = config.train.selection
    except:
        selection = 'best'
    group_postfix += f'_sel-{selection}'

    if augment:
        group_postfix += f'_{ckpt_prefix}_thres{str(thres)}'
        
        if config.augment.segment_flag:
            group_postfix += f'_seg{config.data.thres}'

    predictor_name = config.predictor.target

    group_name = f'{predictor_name}_{data_name}_{group_postfix}'
    if args.remove_dup:
        group_name = 'rev_dup_' + group_name
    if args.use_val_edges:
        group_name = 'use_val_' + group_name
    if args.train_guidance:
        guidance_type = config.augment.guidance_config.diffusion.guidance_type
        group_name = f'{guidance_type}_guide_' + group_name
    if len(args.prefix) > 0:
        group_name = args.prefix + '_' + group_name
    
    if (hasattr(config.augment.sample, 'inpaint_every_step') 
        and not config.augment.sample.inpaint_every_step):

        group_name = 'last_only_' + group_name

    # project_postfix = args.project_postfix
    # if project_postfix is None:
    #     project_postfix = ''

    datetime_now = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    exp_name = f'{group_name}-r.{args.seed}-{datetime_now}'
    wandb.init(
        # project=f'AdjDiff_LinkPredNCN{project_postfix}',
        project=f'AdjDiff_LinkPredNCNVer3',
        group=group_name,
        name=exp_name, 
        config=dict(config),
    )
    # Init: patience=10, eval_steps=5
    # Ver1: patience=100, eval_steps=1
    # Ver2: choose the setting that has better gain
    # Ver3: tune collab
    run(config, args.seed, args.augment, args.use_val_edges, args.remove_dup, args.train_guidance)