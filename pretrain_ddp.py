import os
import time
import wandb
import numpy as np
from pprint import pformat
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from parsers import Parser, get_config
from utils.loader import (
    load_seed, load_device, load_ema, load_data,
    load_diffusion_model_optim, load_evaluater,
    load_denoise_model, load_diffusion_backbone
)
from utils.logger import set_log
from dataset.misc import batched_to_list
from dataset.loader import MultiEpochsPYGDataLoader

torch.set_num_threads(16)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.set_sharing_strategy('file_system')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_dist_dataloader(dataset, batch_size, rank, world_size, shuffle=False):
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)
    dataloader = MultiEpochsPYGDataLoader(dataset, shuffle=False, sampler=sampler, 
                                          batch_size=batch_size, pin_memory=True)
    return dataloader

def train(rank, world_size, config, exp_name='', ckpt_path=None):
    log_folder_name, log_dir, ckpt_dir = set_log(config)
    seed = load_seed(config.train.seed)

    config.exp_name = exp_name
    ckpt_name = f'{exp_name}'

    num_epochs = config.train.num_epochs
    grad_norm = config.train.grad_norm
    save_interval = config.train.save_interval
    eval_interval = config.train.eval_interval
    sample_from_empty = config.sample.sample_from_empty

    if hasattr(config.data, 'cluster_path'):
        cluster_path = config.data.cluster_path
    else:
        cluster_path = None
    dataset = load_data(config.data, return_loader=False, cluster_path=cluster_path)

    if config.data.target == 'network_repository':
        train_dataset = dataset.get_dataloader(return_dataset=True, **config.data)
    else:
        try:
            train_dataset = dataset.get_split_dataset('train')
        except:
            train_dataset = dataset

    batch_size = config.data.batch_size
    train_loader = get_dist_dataloader(
        train_dataset, batch_size=batch_size, rank=rank, world_size=world_size, shuffle=True
    )

    try:
        valid_dataset = dataset.get_split_dataset('valid')
        valid_loader = get_dist_dataloader(
            valid_dataset, batch_size=batch_size, rank=rank, world_size=world_size, shuffle=False
        )
    except:
        valid_loader = None

    try:
        test_dataset = dataset.get_split_dataset('test')
        test_loader = get_dist_dataloader(
            test_dataset, batch_size=batch_size, rank=rank, world_size=world_size, shuffle=False
        )

        evaluater = load_evaluater(
            test_dataset, exp_name=exp_name, device=f'cuda:{rank}'
        )
    except:
        test_loader = None
    
    if hasattr(config.data, 'cluster'):
        config.model.max_degree = dataset.max_degrees[config.data.cluster]
    else:
        config.model.max_degree = dataset.max_degree

    try:
        config.model.max_degrees = dataset.max_degrees
    except:
        pass

    if config.model.target == 'GuidedGNN':
        config.model['num_classes'] = dataset.kmeans.n_clusters

    model_params, diffusion_params, training_params = config.model, config.diffusion, config.train
    model_params.num_timesteps = diffusion_params.num_timesteps
    denoise_model = load_denoise_model(model_params)
    model = load_diffusion_backbone(diffusion_params, denoise_model).cuda(rank)

    if ckpt_path is not None:
        ckpt_state_dict = torch.load(ckpt_path, map_location=f'cuda:{rank}')['model_state_dict']
        if 'module.' in list(ckpt_state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
            ckpt_state_dict = {k[7:]: v for k, v in ckpt_state_dict.items()}
        model.load_state_dict(ckpt_state_dict)
        del ckpt_state_dict

    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        print(f'Num of parameters: {(num_params / 1e6):.4f}M')
        print('\033[91m' + f'{ckpt_name}' + '\033[0m')

    setup(rank, world_size)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.lr, 
                                 weight_decay=training_params.weight_decay)

    # -------- Training --------
    for epoch in (pbar := trange(0, (num_epochs), desc = '[Epoch]', position = 1, leave=False)):

        train_loss = []
        valid_loss = []
        test_metrics = {'degree_mmd': [], 'spectral_mmd': [], 'clustering_mmd': [], 'orbits_mmd': []}

        t_start = time.time()

        model.train()

        for _, train_bdata in enumerate(train_loader):

            optimizer.zero_grad()
            train_bdata = train_bdata.cuda(rank)
            loss = model(train_bdata)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss.append(loss.item())

        if valid_loader is not None:
            with torch.no_grad():
                model.eval()
                for _, valid_bdata in enumerate(valid_loader):   
                    valid_bdata = valid_bdata.cuda(rank)
                    valid_loss.append(model(valid_bdata).item())

        if (epoch + 1) % eval_interval == 0 and test_loader is not None:
            test_stats = test(rank, model, evaluater, test_loader, sample_from_empty)
            for k in test_stats.keys():
                test_metrics[k].append(test_stats[k])

        log_dict = {
            'epoch': epoch, 'time': time.time()-t_start, 'train_loss': np.mean(train_loss),
        }
        tqdm_log = f"[EPOCH {epoch+1:04d}] | train loss: {log_dict['train_loss']:.3e}"

        if valid_loader is not None:
            log_dict['valid_loss'] = np.mean(valid_loss)
            tqdm_log += f" | valid loss: {log_dict['valid_loss']:.3e}"

        if (epoch + 1) % eval_interval == 0 and test_loader is not None:
            for k in test_metrics.keys():
                log_dict[k] = np.mean(test_metrics[k])

        # wandb.log(log_dict)
        pbar.set_description(tqdm_log)
        if epoch % eval_interval == eval_interval - 1:
            tqdm.write(tqdm_log)

        # -------- Save checkpoints --------
        if rank == 0 and epoch % save_interval == save_interval - 1:
            save_name = f'_{epoch+1}' if epoch < num_epochs - 1 else ''
            torch.save({
                'model_config': config,
                'model_state_dict': model.state_dict(), 
                }, f'{ckpt_dir}/{ckpt_name + save_name}.pth')

def test(rank, model, evaluater, test_loader=None, sample_from_empty=False):
    # -------- Sampling --------
    model.eval()
    if True:  # self.config.diffusion.target == 'binary':
        if sample_from_empty:
            generated_data_list = model.sample(num_samples=10).cpu().to_data_list()
        else:
            assert test_loader is not None
            generated_data_list = []
            for _, bdata in tqdm(enumerate(test_loader)):
                bdata = bdata.cuda(rank)
                bdata = model.sample(bdata).cpu()
                generated_data_list.extend(batched_to_list(bdata))

    metrics = evaluater(generated_data_list)
    print(metrics)
    return metrics

def main(config, exp_name='', ckpt_path=None):
    world_size = torch.cuda.device_count()
    # mp.set_start_method('fork')
    mp.spawn(train, args=(world_size, config, exp_name, ckpt_path), nprocs=world_size, join=True)
    # rank = 0
    # train(rank, world_size, config, dataset, exp_name)


if __name__ == '__main__':
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args, unknown = Parser().parse()
    cli = OmegaConf.from_dotlist(unknown)
    config = get_config(args.config, args.seed)
    config = OmegaConf.merge(config, cli)
    print(pformat(vars(config)))

    # -------- Train --------
    seed = config.train.seed
    exp_name = f'{args.prefix}-r.{seed}-{ts}'
    diff_type = config.diffusion.target

    ckpt_path = None
    if hasattr(config.diffusion, 'ckpt_path'):
        ckpt_path = config.diffusion.ckpt_path

    main(config, exp_name, ckpt_path)