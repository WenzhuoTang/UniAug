# TODO: change to link_pred & graph classification
# TODO: impainting, pure generation, augmentation



import math
import time
import wandb
import functools
from tqdm import tqdm
from pprint import pformat
from datetime import datetime
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from parsers import Parser, get_config
from dataset.loader import MultiEpochsPYGDataLoader
from utils import (
    AverageMeter, validate, print_info, init_weights, load_generator, 
    ImbalancedSampler, build_augmentation_dataset, set_seed,
    dict_of_dicts_to_dict, flatten_dict, unflatten_dict
)
from utils.loader import load_device, load_data, load_downstream_model

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')
torch.set_num_threads(16)
# torch.multiprocessing.set_sharing_strategy('file_system')

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    # num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(1e-2, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def train(model, train_loaders, optimizer, scheduler, epoch, epochs, steps, device='cuda', p_bar=None, prof=None, 
          enable_scheduler=True):
    criterion = cls_criterion
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    loss_list = []
    for batch_idx in range(steps):
        if prof is not None:  # pytorch profiler
            prof.step()
        end = time.time()
        model.zero_grad()
        try:
            batch_labeled = next(train_loaders['labeled_iter'])
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = next(train_loaders['labeled_iter'])
        batch_labeled =  batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        is_labeled = targets == targets
        if batch_labeled.x.shape[0] == 1 or batch_labeled.batch[-1] == 0:
            continue
        else:
            pred_labeled = model(batch_labeled)[0]
            Losses = criterion(pred_labeled.view(targets.size()).to(torch.float32)[is_labeled], targets[is_labeled])
            loss = Losses.mean()
        loss.backward()
        optimizer.step()
        if enable_scheduler:
            scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        loss_list.append(losses.avg)
        if p_bar is not None:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=epochs,
                    batch=batch_idx + 1,
                    iter=steps,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    )
            )
            p_bar.update()
    if p_bar is not None:
        p_bar.close()
    return train_loaders, sum(loss_list) / len(loss_list)

def get_augmented(model, generator, loader, use_sudo_label=False, device='cuda:0', sde_x_config={},
                  sde_adj_config={}, batch_size=64, shuffle=False, num_workers=16, augment_all_graphs=False,
                  augment_kwargs={}, return_loader=True):
    new_dataset = build_augmentation_dataset(model, generator, loader, use_sudo_label=use_sudo_label, device=device, 
                                             sde_x_config=sde_x_config, sde_adj_config=sde_adj_config,
                                             augment_all_graphs=augment_all_graphs, **augment_kwargs)
    if return_loader:
        # return DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return MultiEpochsPYGDataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        return new_dataset

def main(config, split_dict, task='NC', data_name='cora', seed=1, augment=True, num_workers=0, 
         sweep=False, prof_flag=False, extreme_mode=True, **kwargs):
    if sweep:
        wandb.run=None
        run = wandb.init(group=f'downstream_{task}_{data_name}_{config.data.subgraph_type}_aug{str(augment)}')
        sweep_config = unflatten_dict(wandb.config)
        config = OmegaConf.merge(config, dict(sweep_config))

    set_seed(seed, extreme_mode=extreme_mode)

    lr = config.train.lr
    wdecay = config.train.wdecay
    epochs = config.train.epochs
    patience = config.train.patience
    batch_size = config.data.batch_size
    val_int = config.train.val_int if config.train.val_int is not None else 1 

    start = config.augment.start
    iteration = config.augment.iteration
    augment_valid = config.augment.augment_valid
    augment_test = config.augment.augment_test

    device = load_device()
    if isinstance(device, list):
        device = f'cuda:{device[0]}'
    labeled_dataset = load_data(config, return_loader=False)

    loaders = {}
    for k in split_dict.keys():
        shuffle = True if 'train' in k else False
        split = [f'{data_name}-{task}-{s}' for s in split_dict[k]]
        loaders[k] = labeled_dataset.get_dataloader(split=split, shuffle=shuffle, **config.data)

    train_loaders = {
        'labeled_iter': iter(loaders['train']),
        'labeled_trainloader': loaders['train'],
    }

    label_split_idx = labeled_dataset.splits
    num_trained = len(label_split_idx[f"{data_name}-{task}-{split_dict['train'][0]}"])
    num_trained_init = num_trained
    steps = num_trained // batch_size + 1
    strategy = config.augment.strategy

    config.model.target = 'GNN' if config.model.target is None else config.model.target
    config.model.input_dim = labeled_dataset.x.shape[1]
    if config.model.target == 'GNN':
        config.model.num_tasks = labeled_dataset.y.shape[1]
    model_type = config.model.gnn_type
    if config.model.target == 'DGCNN':
        k = config.model.k
        if k <= 1:  # Transform percentile to number.
            if False:  # train_dataset is None:
                k = 30
            else:
                if False:  # dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = loaders['train'].dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        config.model.k = k
    model = load_downstream_model(config.model).to(device)
    if augment:
        generator = load_generator(device, path=config.augment.ckpt_path)
    init_weights(model, config.train.initw_name, init_gain=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)

    best_results = {}
    if task == 'NC':
        metric = 'acc' # 'macro_auc'
    elif task == 'LP':
        metric = 'hits@100' # 'mrr'
    else:
        raise NotImplementedError(f'Unsupported task {task}')

    loaders_to_aug = loaders.copy()
    loaders_to_aug['train'] = train_loaders['labeled_trainloader']
    for epoch in (p_bar := tqdm(range(epochs))):
        if prof_flag and epoch % (config.augment.start + 1) == 0:
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{config.model.gnn_type}-augment_{str(augment)}'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
            ) as prof:
                train_loaders, loss = train(model, train_loaders, optimizer, scheduler, epoch, epochs, steps, 
                                            device, prof=prof)
            print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        else:
            train_loaders, loss = train(model, train_loaders, optimizer, scheduler, epoch, epochs, steps, device)
        if epoch % val_int == 0:
            results = {}
            results['train'] = validate(task, model, loaders['train'], device, model_type)
            results['valid'] = validate(task, model, loaders['valid'], device, model_type)
            results['test'] = validate(task, model, loaders['test'], device, model_type)

        if augment and epoch >= start and epoch % iteration == 0:
            train_loaders['labeled_trainloader'] = get_augmented(
                model, generator, loaders_to_aug['train'], use_sudo_label=False, device=device, 
                sde_x_config=config.sde.x, sde_adj_config=config.sde.adj, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, augment_kwargs=config.augment
            )
            if augment_valid:
                loaders['valid'] = get_augmented(
                    model, generator, loaders_to_aug['valid'], use_sudo_label=True, device=device, sde_x_config=config.sde.x,
                    sde_adj_config=config.sde.adj, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, augment_all_graphs=True, augment_kwargs=config.augment
                )
            if augment_test:
                loaders['test'] = get_augmented(
                    model, generator, loaders_to_aug['test'], use_sudo_label=True, device=device, sde_x_config=config.sde.x,
                    sde_adj_config=config.sde.adj, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, augment_all_graphs=True, augment_kwargs=config.augment
                )

            num_trained = len(train_loaders['labeled_trainloader'].dataset)
            steps = num_trained // batch_size + 1
            if strategy.split('_')[-1] == 'accumulate':
                loaders_to_aug['train'] = train_loaders['labeled_trainloader']
                if augment_valid:
                    loaders_to_aug['valid'] = loaders['valid']
                if augment_test:
                    loaders_to_aug['test'] = loaders['test']
            if num_trained > num_trained_init * 2:
                strategy = 'replace' + '_' + strategy.split('_')[-1]
            config.augment.strategy = strategy

        # update_test = True # False
        # if epoch != 0 and task == 'NC' and results['valid'][metric] >  best_results['valid'][metric]:
        #     update_test = True
        # elif epoch != 0 and task == 'LP' and results['valid'][metric] >  best_results['valid'][metric]:
        #     update_test = True
        # if update_test or epoch == 0:
        if epoch % val_int == 0:
            if epoch != 0 and results['valid'][metric] < best_results['valid'][metric]:
            # if epoch != 0 and results['valid'][metric] <= best_results['valid'][metric]:
                if epoch > 30:  # 30 
                    cnt_wait += 1
                    if cnt_wait > patience:
                        break
            else:
                best_results = results
                cnt_wait = 0
                best_epoch = epoch

            wandb.log(dict_of_dicts_to_dict(results))
            if metric not in results['train']:
                results['train'][metric] = 0
            p_bar.set_description("Epoch: {e}/{es:4}. Loss: {l:.6f}. Train metric: {trm:.4f}. Validation metric: {vam:.4f}. Test metric: {tem:.4f}. ".format(
                e=epoch + 1,
                es=epochs,
                l=loss,
                trm=results['train'][metric],
                vam=results['valid'][metric],
                tem=results['test'][metric],
            ))

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print_info('train_best', best_results['train'])
    print_info('valid_best', best_results['valid'])
    print_info('test_final', best_results['test'])
    results = {
        'train_best': best_results['train'],
        'valid_best': best_results['valid'],
        'test_final': best_results['test'],
    }
    wandb.log(dict_of_dicts_to_dict(results))
    return results

def run(prefix, task, data_name, config, split_dict, augment, seed=0, 
        extreme_mode=True, sweep=False, sweep_id=None, prof_flag=False):
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    exp_name = f'{prefix}-{datetime_now}'
    if not sweep:
        _ = wandb.init(
            project='GraphDiff',
            group=f'downstream_{task}_{data_name}_{config.data.subgraph_type}_aug{str(augment)}',
            name=exp_name, 
            config=dict(config),
        )
        results = main(config, split_dict, task, data_name, seed, augment=augment,
                       prof_flag=prof_flag, extreme_mode=extreme_mode)
        return results
    else:
        
        params = {
            'model':{
                'emb_dim': {'values': [32, 64, 128, 256, 512]},
                'num_layers': {'values': [1, 2, 3, 4]},
                # 'subset': {'values': [True, False]},
                'subset': {'values': [False]},
                'JK': {'values': ['last', 'sum']},  # stack
                'gnn_type': {'values': ['gcn']},  # gcn | gin | sage
                # 'graph_pooling': {'values': ['mean', 'max', 'sum']},
                'drop_ratio': {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                'act': {'values': ['relu', 'tanh', 'sigmoid']},
                # LP
                'predictor_type': {'values': ['cnn', 'ncn', 'ncnc']},
                'pred_edp': {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                'beta': {'values': [0.25, 0.5, 0.75, 1]},
                'alpha': {'values': [0.25, 0.5, 0.75, 1]},
                'use_feature': {'values': [True, False]},
            },
            'train':{
                'lr': {'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
                'wdecay': {'values': [0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
                # 'lr': {'values': [1e-4, 5e-4, 1e-3, 5e-3]},
                # 'wdecay': {'values': [0., 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]},
            },
            'augment':{
                'n_negative': {'values': [1, 10, 50, 100]},
                'out_steps': {'values': [1, 10, 50, 100, 500]},
                'topk': {'values': [5, 10, 50, 128]},
                'start': {'values': [10, 20, 50, 100]},
                'iteration': {'values': [1, 5, 10, 20]},
                'strategy': {'values': ['replace_once', 'add_once', 
                                        'replace_accumulate', 'add_accumulate']},
                'snr': {'values': [0, 0.25, 0.5, 0.75, 0.9]},
                'scale_eps': {'values': [0, 0.25, 0.5, 0.75, 0.9]},
                'perturb_ratio': {'values': [0., 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]},
                'n_steps': {'values': [1, 5, 10, 20]},
                'aug_adj': {'values': [True, False]},
                'feat_to_compare': {'values': ['sum', 'extract']},
                'augment_valid': {'values': [True, False]},
                'augment_test': {'values': [True, False]},
            }
        }
        if not augment:
            params.pop('augment')
        params = flatten_dict(params, stop_keys=['values'])

        sweep_configuration = {
            'method': 'bayes',
            'name': exp_name,
            'metric': {
                'goal': 'maximize', 
                'name': 'test_final-acc' if task == 'NC' else 'test_final-hits@100'
                },
            'parameters': params,
            # 'run_cap': 1000,
        }
        _sweep_id = wandb.sweep(sweep=sweep_configuration, project='GraphDiff')
        sweep_id = sweep_id if sweep_id is not None else _sweep_id
        print(f'sweep_id: {sweep_id}')
        wandb.agent(sweep_id, count=10000, 
                    function=functools.partial(main, config, split_dict, task, data_name, seed, 
                                               augment=augment, sweep=sweep, extreme_mode=extreme_mode))
        return None

if __name__ == '__main__':
    # TODO: update argument; implement GNN for NC and LP 
    # args = get_args()
    # config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
    args, unknown = Parser().parse()
    cli = OmegaConf.from_dotlist(unknown)
    config = get_config(args.config, args.seed)
    config = OmegaConf.merge(config, cli)
    if args.orig_feat:
        config.data.feature_types = None 
        config.data.stru_feat_principle = None
    if args.full_subgraph:
        config.data.max_node_num = None

    print(pformat(dict(config)))
    for k in config.keys():
        for arg, value in config[k].items():
            setattr(args, arg, value)

    task = config.data.task
    data_name = config.data.data_name
    if task == 'LP':
        # XXX: for SEALDataset
        split_list = ['train_pos', 'train_neg', 'valid_pos', 'valid_neg', 'test_pos', 'test_neg']
        split_dict = {
            'train': ['train_pos', 'train_neg'], 
            'valid': ['valid_pos', 'valid_neg'], 
            'test': ['test_pos', 'test_neg'], 
        }
        # metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mrr', 'auc', 'ap']
    elif task == 'NC':
        split_list = ['train', 'test', 'valid']
        split_dict = {split: [split] for split in split_list}
        metric_list = ['acc', 'macro_acc', 'macro_f1', 'macro_auc']

    if args.trails > 0:
        results = {m: [] for m in metric_list}
        for i in range(args.trails):
            temp_results = run(args.prefix, task, data_name, config, split_dict,
                               args.augment, seed=i, extreme_mode=True)
            for m in metric_list:
                results[m].append(temp_results['test_final'][m])
        for m in metric_list:
            print(f'{m}: {np.mean(results[m])} +/- {np.std(results[m])}')
    else:
        results = run(args.prefix, task, data_name, config, split_dict, args.augment, 
                      seed=args.seed, extreme_mode=True, sweep=args.sweep, sweep_id=args.sweep_id,
                      prof_flag=args.profile)
    # print(pformat(dict(config)))
    # logger = get_logger(__name__, logfile=None)
    # print(args)
    # results = {}
    # if args.trails > 0:
    #     for exp_num in range(args.trails):
    #         set_seed(exp_num)
    #         args.exp_num = exp_num
    #         results['train'], results['valid'], results['test'] = main(args)
    #         exp_result_temp = {'train': results['train'], 'valid': results['valid'], 'test': results['test']}
    #         if exp_num == 0:
    #             for metric in results['train'].keys():
    #                 results[f'train_{metric}'] = []
    #                 results[f'valid_{metric}'] = []
    #                 results[f'test_{metric}'] = []
    #         for name in ['train', 'test', 'valid']:
    #             if args.task_type in 'regression':
    #                 metric_list = ['rmse', 'r2','mae','mse']
    #             else:
    #                 metric_list = ['auc']
    #             for metric in metric_list:
    #                 results[f'{name}_{metric}'].append(exp_result_temp[name][metric])
    #         for mode, nums in results.items():
    #             print('{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums))