import time
import wandb
import numpy as np
import os.path as osp
from tqdm import tqdm, trange
from pprint import pformat
from omegaconf import OmegaConf
from ogb.graphproppred import Evaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import degree, add_remaining_self_loops
from torch_geometric.loader import DataLoader

from parsers import Parser, get_config
from dataset.loader import get_batched_datalist, MultiEpochsPYGDataLoader
from dataset.misc import batched_to_list
from dataset.property import get_properties, split_by_property
from utils import set_seed
from utils.loader import (
    load_data, 
    load_sampler,
    load_downstream_model,
    load_diffusion_guidance_optim,
)

torch.set_num_threads(2)

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

### Early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, mode="max"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        if mode == "max":
            self.best_score = -np.Inf
            self.check_func = lambda x, y: x >= y
        else:
            self.best_score = np.inf
            self.check_func = lambda x, y: x <= y

    def __call__(self, score):
        if self.check_func(score, self.best_score + self.delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

### Training Loop
def train(model, device, loader, optimizer, task_type='classification'):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

### Evaluation Loop
@torch.no_grad()
def eval(model, device, loader, evaluator=None, task_type='classification'):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if task_type == 'classification':
        results = evaluator.eval(input_dict)
    else:
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        results = {'rmse': rmse, 'mse': mse, 'mae': mae}
    
    return results

### Main function for graph prediction
def run_graph_pred(model, train_loader, valid_loader, test_loader, device='cuda:0', epochs=1000,
                   lr=1e-3, step_size=50, gamma=0.5, patience=500, task_type='classification', 
                   metric='rocauc', weight_decay=0., eval_start=0, selection='best', **kwargs):

    model.to(device)
    # model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    scheduler = StepLR(optimizer, step_size, gamma)
    # scheduler = None

    mode = 'max' if metric == 'rocauc' else 'min'
    evaluator = Evaluator(data_name) if task_type == 'classification' else None
    early_stopper = EarlyStopping(patience=patience, verbose=False, mode=mode)

    all_results = {}
    for epoch in (pbar := trange(0, (epochs), desc = '[Epoch]', position = 1, leave=False)):
        train(model, device, train_loader, optimizer, task_type)

        if scheduler is not None:
            scheduler.step()

        if epoch >= eval_start:
            valid_results = eval(model, device, valid_loader, evaluator, task_type)
            test_results = eval(model, device, test_loader, evaluator, task_type)
            valid_metric = valid_results[metric]

            results = {}
            for k in valid_results.keys():
                results[f'valid_{metric}'] = valid_results[k]
                results[f'test_{metric}'] = test_results[k]

            wandb.log(results)

            for k in results.keys():
                if k not in all_results.keys():
                    all_results[k] = []
                all_results[k].append(results[k])

            msg = f"Epoch: {epoch}, {', '.join([f'{k}: {results[k]}' for k in sorted(results.keys())])}"
            pbar.set_description(msg)

            if early_stopper(valid_metric):
                break

            if early_stopper.counter == 0:
                tqdm.write(msg)

    temp = np.array(all_results[f'valid_{metric}'])
    if mode == 'max':
        best_idx = np.where(temp == temp.max())[0][-1]
    else:
        best_idx = np.where(temp == temp.min())[0][-1]

    final_results = {}
    for k in all_results.keys():
        s, m = k.split('_')[0], k.split('_')[1]
        if s == 'test':
            temp = np.array(all_results[f'valid_{m}'])

            if selection == 'best':
                if mode == 'max':
                    best_idx = np.where(temp == temp.max())[0][-1]
                else:
                    best_idx = np.where(temp == temp.min())[0][-1]

            elif selection == 'last':
                best_idx = -1
            else:
                raise ValueError

            final_results[f'final_{m}'] = all_results[f'test_{m}'][best_idx]
            tqdm.write(f"Best epoch: {best_idx}, final {m}: {final_results[f'final_{m}']}")

    wandb.log(final_results)

    return final_results


def main_fold(config, train_dataset, valid_dataset, test_dataset, device='cuda:0', seed=10,
              augment=False, train_guidance=False, subset_ratio=None, neg_guide=False):
    
    """
    Main function to perform training, validation, and testing of a graph prediction model with optional data augmentation.
    Args:
        augment (bool, optional): Flag to indicate whether to perform data augmentation. Default is False.
        train_guidance (bool, optional): Flag to indicate whether to train a guidance model. Default is False.
        subset_ratio (float, optional): Ratio of the subset of the training data to use. Default is None.
        neg_guide (bool, optional): Flag to indicate whether to use negative guidance. Default is False.
    Returns:
        dict: Results on the test dataset.
    """
    data_params, model_params, train_params = config.data, config.model, config.train
    data_name = data_params.data_name

    model_params.num_tasks = train_dataset.y.shape[1]
    model = load_downstream_model(model_params).to(device)

    if augment:
        augment_params = config.augment        
        ckpt_path = augment_params.ckpt_path

        train_dataset = [train_dataset.get(i) for i in train_dataset.indices()]
        data_list = train_dataset.copy()

        if subset_ratio is not None and subset_ratio < 1:
            from sklearn.model_selection import train_test_split

            stratify_labels = torch.cat([data.y for data in data_list])
            _, subset_idx = train_test_split(np.arange(len(data_list)), test_size=subset_ratio,
                                             random_state=seed, stratify=stratify_labels)
            data_list = [data_list[i] for i in subset_idx]

        if train_guidance:
            guidance_config = augment_params.guidance_config

            ########################### temp feature of negative guidance
            if neg_guide:
                if data_name in ['Enzymes', 'Proteins']:
                    neg_data_name = 'IMDB_multi'
                elif data_name in ['IMDB_binary', 'IMDB_multi']:
                    neg_data_name = 'Enzymes'

                print(f'Ablation study of negative transfer')
                data_params.data_name = neg_data_name
                neg_dataset = load_data(data_params, return_loader=False, cluster_path=cluster_path)
                guidance_data_list = [neg_dataset.get(i) for i in neg_dataset.indices()]
                print(f'Training guidance head with {neg_dataset.data_name}')

            else:
                guidance_data_list = data_list

            if guidance_config.diffusion.guidance_type == 'graph_class':
                guidance_config.guidance.output_dim = guidance_data_list[0].labels.shape[1]
            
            elif guidance_config.diffusion.guidance_type in ['graph_binary', 'graph_regression']:
                guidance_config.guidance.output_dim = guidance_data_list[0].y.shape[1]

            elif guidance_config.diffusion.guidance_type == 'graph_prop':
                guidance_data_list, property_attr, _, _ = get_properties(guidance_data_list, attr_name='prop_attr')
                guidance_config.guidance.output_dim = property_attr.shape[1]

            freeze_model = guidance_config.diffusion.freeze_model
            guidance_config, sampler, sampler_optim, sampler_sched = load_diffusion_guidance_optim(
                guidance_config, device, ckpt_path, freeze_model=freeze_model
            )

            ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
            guidance_ckpt_name = f"{ckpt_path.split('/')[-1].split('-')[0]}-guidance_{data_name}-r.{seed}-{ts}.pth"
            guidance_ckpt_dir = osp.join(*ckpt_path.split('/')[:-1])
            guidance_ckpt_path = osp.join(guidance_ckpt_dir, guidance_ckpt_name)

            num_epochs, batch_size = guidance_config.num_epochs, int(guidance_config.batch_size)
            dataloader = MultiEpochsPYGDataLoader(guidance_data_list, batch_size=batch_size, shuffle=True)

            print('Training guidance model ...')
            for epoch in (pbar := trange(0, (num_epochs), desc = '[Epoch]', position = 1, leave=False)):
                losses = []
                for _, bdata in enumerate(dataloader):
                    bdata = bdata.to(device)
                    loss = sampler(bdata)
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
            
            torch.cuda.empty_cache()

        else:
            sampler = load_sampler(ckpt_path, device=device)

        def sample_graphs(data_list, nodes_max=None, edges_max=None, batch_size=1, sample_params={},
                          num_repeats=1, keys_to_keep=None):

            # get kmeans labels for self-guided diffusion
            kmeans, _, _, scaler_dict = torch.load(cluster_path)
            data_list, property_attr, _, _ = get_properties(
                data_list, scaler_dict=scaler_dict, nodes_and_edges=True
            )
            _, kmeans_labels = split_by_property(property_attr, kmeans=kmeans)
            for i in range(len(data_list)):
                data_list[i].kmeans_labels = torch.from_numpy(
                    kmeans_labels[i].repeat(data_list[i].num_nodes)
                )

            if nodes_max is not None or edges_max is not None:
                dataloader = get_batched_datalist(data_list, nodes_max=nodes_max, edges_max=edges_max)
                print(len(dataloader))
            else:
                batch_size = int(augment_params.batch_size)
                dataloader = MultiEpochsPYGDataLoader(data_list, batch_size=batch_size, shuffle=False)

            new_data_list = []
            for i in range(num_repeats):
                print(f'Augmenting... repeat {i + 1} / {num_repeats}')
                for _, bdata in tqdm(enumerate(dataloader)):
                    bdata = bdata.to(device)
                    bdata = sampler.sample(
                        bdata, device=device, **sample_params
                    ).cpu()
                    new_data_list.extend(batched_to_list(bdata, keys_to_keep=keys_to_keep))
            
            for i in range(len(new_data_list)):
                if data_name in ['Reddit_binary', 'Reddit_multi_5k', 'Reddit_multi_12k',
                                 'IMDB_binary', 'IMDB_multi', 'Collab']:
                    new_data_list[i].x = degree(new_data_list[i].edge_index[0]).unsqueeze(-1)

                new_data_list[i].edge_index, _ = add_remaining_self_loops(new_data_list[i].edge_index)

            return new_data_list

        def remove_attributes_from_dataset(dataset, keys_to_remove):
            if len(keys_to_remove) > 0:
                for data in dataset: 
                    for k in keys_to_remove:
                        data.pop(k, None)

            return dataset

        try:
            nodes_max = augment_params.nodes_max
        except:
            nodes_max = None
        
        try:
            edges_max = augment_params.edges_max
        except:
            edges_max = None

        batch_size = int(augment_params.batch_size)
        sample_params = OmegaConf.to_container(augment_params.sample)
        num_repeats = int(augment_params.num_repeats)
        keys_to_keep = train_dataset[0].keys()

        new_data_list = sample_graphs(data_list, nodes_max=nodes_max, edges_max=edges_max,
                                      batch_size=batch_size, sample_params=sample_params,
                                      num_repeats=num_repeats, keys_to_keep=keys_to_keep)

        replace_flag = augment_params.replace_flag
        keys_to_remove = list(set(train_dataset[0].keys()) - set(new_data_list[0].keys()))
        if replace_flag:
            train_dataset = new_data_list
        else:
            train_dataset = train_dataset + new_data_list
        train_dataset = remove_attributes_from_dataset(train_dataset, keys_to_remove)

        ## sanity check: remove empty graphs
        num_nodes = np.array([g.num_nodes for g in train_dataset])
        idx2remove = np.where(num_nodes == 0)[0]
        train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i not in idx2remove]

        keys_to_keep = train_dataset[0].keys()
        augment_valid, augment_test = augment_params.augment_valid, augment_params.augment_test
        if augment_valid:
            data_list = [valid_dataset.get(i) for i in valid_dataset.indices()] 
            valid_dataset = sample_graphs(data_list, nodes_max=nodes_max, edges_max=edges_max,
                                          batch_size=batch_size, sample_params=sample_params,
                                          num_repeats=1, keys_to_keep=keys_to_keep)
        else:
            valid_dataset = [valid_dataset.get(i) for i in valid_dataset.indices()]
        valid_dataset = remove_attributes_from_dataset(valid_dataset, keys_to_remove)

        if augment_test:
            data_list = [test_dataset.get(i) for i in test_dataset.indices()] 
            test_dataset = sample_graphs(data_list, nodes_max=nodes_max, edges_max=edges_max,
                                          batch_size=batch_size, sample_params=sample_params,
                                          num_repeats=1, keys_to_keep=keys_to_keep)
        else:
            test_dataset = [test_dataset.get(i) for i in test_dataset.indices()]
        test_dataset = remove_attributes_from_dataset(test_dataset, keys_to_remove)

        del sampler, data_list, new_data_list
        torch.cuda.empty_cache()
        print('Using augmented view to train')
    
    # else:
    #     train_dataset = [train_dataset.get(i) for i in train_dataset.indices()]
    #     valid_dataset = [valid_dataset.get(i) for i in valid_dataset.indices()]
    #     test_dataset = [test_dataset.get(i) for i in test_dataset.indices()]

    batch_size = data_params.batch_size
    train_loader = MultiEpochsPYGDataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = MultiEpochsPYGDataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    test_loader = MultiEpochsPYGDataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    results = run_graph_pred(model, train_loader, valid_loader, test_loader, device, **train_params)
    return results

### Main Entry function
def main(config, seed=10, augment=False, train_guidance=False, neg_guide=False):
    set_seed(seed)

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data_params = config.data
    try:
        subset_ratio = float(config.augment.subset_ratio)
    except:
        subset_ratio = None

    dataset = load_data(data_params, return_loader=False, cluster_path=cluster_path)

    splits = dataset.splits
    splits = [splits] if not isinstance(splits, list) else splits

    all_results = {}
    for spl in splits:
        train_dataset = dataset[spl['train']]
        valid_dataset = dataset[spl['valid']]
        test_dataset = dataset[spl['test']]

        results = main_fold(config, train_dataset, valid_dataset, test_dataset, device=device, 
                            seed=seed, augment=augment, train_guidance=train_guidance, 
                            subset_ratio=subset_ratio, neg_guide=neg_guide)

        for k in results.keys():
            if k not in all_results.keys():
                all_results[k] = []
            all_results[k].append(results[k])

    for k in all_results.keys():
        print("##################")
        print(f"{k}: {np.mean(all_results[k])} +/- {np.std(all_results[k])}")
        print("##################")

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
    else:
        cluster_path = 'data/misc/full_network_repository/processed/entr10_dens0.1_denm50_degm3_degv3_node4000_edge50000-feat_none-ext_github_stargazers-all_10clusters_ne.pt'

    ckpt_prefix = ckpt_prefix.split('_dnnm-')[0]
    ckpt_epochs = ['1000', '3000', '5000', '7000']
    for ep in ckpt_epochs:
        if ep in ckpt_path:
            ckpt_prefix += f'_{ep}'

    thres = args.thres if args.thres is not None else config.augment.sample.thres
    config.augment.sample.thres = thres  # overwrite

    num_repeats = config.augment.num_repeats
    replace_flag = config.augment.replace_flag

    group_postfix = f'aug{str(augment)}'
    try:
        selection = config.train.selection
    except:
        selection = 'best'
    
    if selection != 'best':
        group_postfix += f'_sel-{selection}'

    if augment:
        try:
            subset_ratio = float(config.augment.subset_ratio)
        except:
            subset_ratio = None

        if subset_ratio is not None and subset_ratio < 1:
            group_postfix += f'_sub{subset_ratio}'

        group_postfix += f'_{ckpt_prefix}_thres{str(thres)}_nrep{num_repeats}_repl{str(replace_flag)}'

    group_name = f'{data_name}_{group_postfix}'
    if args.train_guidance:
        guidance_type = config.augment.guidance_config.diffusion.guidance_type
        group_name = f'{guidance_type}_guide_' + group_name

    if len(args.prefix) > 0:
        group_name = args.prefix + '_' + group_name

    if args.neg_guide:
        group_name = 'neg_' + group_name

    edge_feat_flag = config.model.edge_feat
    if edge_feat_flag:
        group_name = 'edge_' + group_name
        
    ### wandb init

    datetime_now = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    exp_name = f'{group_name}-r.{args.seed}-{datetime_now}'
    wandb.init(
        project='AdjDiff_GraphPropPredMol',
        group=group_name,
        name=exp_name, 
        config=dict(config),
    )
    ## Ver: cross-validation
    main(config, args.seed, args.augment, args.train_guidance, args.neg_guide)