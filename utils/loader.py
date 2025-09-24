import os
import random
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.distributed as dist

from models.gnn import GNN, SimpleGCN, GIN, GCNWithPooling, NCNGCN, GINVirtualnode
from models.link_pred import CNLinkPredictor, IncompleteCN1Predictor, MLPLinkPredictor, CFLinkPredictor
from utils.models.ScoreNetwork_A import ScoreNetworkA
from utils.models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from utils.sde import VPSDE, VESDE, subVPSDE
from utils.losses import get_sde_loss_fn
from utils.mmd import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage
from utils.misc import update
from eval.evaluator import GenericGraphEvaluator


# ----------------------------------------
# -------- General loaders --------
# ----------------------------------------
def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device

def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema

def load_data(params, return_loader=False, cluster_path=None):
    from dataset import ( 
        UnlabeledInducedDataset,
        DownstreamInducedDataset,
        SEALDataset,
        UnlabeledSegmentsDataset,
        GenericDataset,
        LinkPredictionDataset,
        NetworkRepositoryDataset,
        GraphPropertyPredictionDataset,
        NodeInducedDataset
    )
    if cluster_path is not None:
        params = OmegaConf.to_container(params)
        kmeans, _, _, scaler_dict = torch.load(cluster_path)
        params['kmeans'] = kmeans
        params['scaler_dict'] = scaler_dict

    assert params['target'] is not None
    if params['target'] == 'downstream_induced':
        dataset = DownstreamInducedDataset(**params)
    elif params['target'] == 'unlabeled_induced':
        dataset = UnlabeledInducedDataset(**params)
    elif params['target'] == 'seal':
        dataset = SEALDataset(**params)
    # elif params['target'] == 'downstream_segments':
    #     dataset = DownstreamSegmentsDataset(**params)
    elif params['target'] == 'unlabeled_segments':
        dataset = UnlabeledSegmentsDataset(**params)
    elif params['target'] == 'generic':
        dataset = GenericDataset(**params)
    elif params['target'] == 'link_prediction':
        dataset = LinkPredictionDataset(**params)
    elif params['target'] == 'network_repository':
        dataset = NetworkRepositoryDataset(**params)
    elif params['target'] == 'graph_prop_pred':
        dataset = GraphPropertyPredictionDataset(**params)
    elif params['target'] == 'node_induced':
        dataset = NodeInducedDataset(**params)
    else:
        raise NotImplementedError(f"Unsupported dataset type {params['target']}")

    if return_loader:
        return dataset.get_dataloader(**params)
    else:
        return dataset

# ----------------------------------------
# -------- Diffusion loaders --------
# ----------------------------------------
def load_diffusion_backbone(params, model):
    if params['target'] == 'bernoulli':
        from diffusion.bernoulli_diffusion import BernoulliDiffusion
        return BernoulliDiffusion(model=model, **params)
    elif params['target'] == 'bernoulli_onehot':
        from diffusion.binary_diffusion import BernoulliOneHotDiffusion
        return BernoulliOneHotDiffusion(model=model, **params)
    elif params['target'] == 'binary':
        from diffusion.discrete_diffusion import BinaryDiffusion
        return BinaryDiffusion(model=model, **params)
    else:
        raise NotImplementedError(f"Unsupported diffusion type {params['target']}")

def load_denoise_model(params, **kwargs):
    from diffusion.models import (
        GNN, GuidedGNN, MSTAGNN, PowerGraphTransformer, TGNN, TGNNDegreeGuided
    )
    if params['target'] == 'GNN':
        model = GNN(**params)
    elif params['target'] == 'GuidedGNN':
        model = GuidedGNN(**params, **kwargs)
    elif params['target'] == 'STA':
        model = MSTAGNN(**params)
    elif params['target'] == 'GT':
        model = PowerGraphTransformer(**params)
    elif params['target'] == 'TGNN':
        model = TGNN(**params)
    elif params['target'] == 'TGNN_degree':
        model = TGNNDegreeGuided(**params)
    else:
        raise NotImplementedError(f"Unsupported denoise model type {params['target']}")
    
    print('-'*100)
    print('Denoisng model loaded!')
    print(model)
    return model

def load_diffusion_params(config):
    return config.model, config.diffusion, config.train

def load_diffusion_model_optim(config, device, **kwargs):
    model_params, diffusion_params, training_params = load_diffusion_params(config)
    model_params.num_timesteps = diffusion_params.num_timesteps
    denoise_model = load_denoise_model(model_params, **kwargs)
    model = load_diffusion_backbone(diffusion_params, denoise_model)
    num_params = sum(param.numel() for param in model.parameters())
    print(f'Num of parameters: {(num_params / 1e6):.4f}M')

    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
            # model = torch.nn.parallel.DistributedDataParallel(model)
        model = model.to(f'cuda:{device[0]}')
        model.device = f'cuda:{device[0]}'
    else:
        model = model.to(device)
        model.device = device
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.lr, 
                                 weight_decay=training_params.weight_decay)
    scheduler = None
    if training_params.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training_params.lr_decay)
    
    return model, optimizer, scheduler

def load_diffusion_from_ckpt(config, state_dict, device):
    diffusion_model = load_diffusion_model_optim(config, device)[0]
    diffusion_model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=device)
            # diffusion_model = torch.nn.parallel.DistributedDataParallel(diffusion_model)
        diffusion_model = diffusion_model.to(f'cuda:{device[0]}')
    else:
        diffusion_model = diffusion_model.to(device)
    return diffusion_model

def load_sampler(path, device):
    device = f'cuda:{device[0]}' if isinstance(device, list) else device
    print('Loading to device: ', device)
    ckpt = torch.load(path, map_location=device)
    print(f'Loading denoising model from {path}')
    sampler = load_diffusion_from_ckpt(ckpt['model_config'], ckpt['model_state_dict'], device)
    print(f'Denoising model loaded!')
    return sampler

def load_evaluater(reference_dataset, exp_name='run', device='cuda:0'):
    return GenericGraphEvaluator(reference_dataset, exp_name=exp_name, device=device)

def load_diffusion_guidance(params, model, guidance_head, ckpt_path=None):
    if params['target'] == 'binary':
        from diffusion.discrete_diffusion import BinaryDiffusionGuidance

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            diffusion_params = ckpt['model_config'].diffusion
            params = update(params, diffusion_params)

        diffusion_model = BinaryDiffusionGuidance(model=model, guidance_head=guidance_head, **params)
        if ckpt_path is not None:
            model_dict = diffusion_model.state_dict()
            ckpt_state_dict = ckpt['model_state_dict']

            if 'module.' in list(ckpt_state_dict.keys())[0]:
                # strip 'module.' at front; for DDP models
                ckpt_state_dict = {k[7:]: v for k, v in ckpt_state_dict.items()}

            loadable_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            ignored_keys = [x for x in list(ckpt_state_dict) if x not in list(loadable_state_dict)]
            not_pretrained_keys = [x for x in list(model_dict) if x not in list(loadable_state_dict)]

            if len(ignored_keys) > 0:
                print(f"Ignored keys: {ignored_keys}")
            if len(not_pretrained_keys) > 0:
                print(f"Not pretrained keys: {not_pretrained_keys}")

            model_dict.update(loadable_state_dict)
            diffusion_model.load_state_dict(model_dict)
        
        return diffusion_model

    else:
        raise NotImplementedError(f"Unsupported diffusion guidance type {params['target']}")
    
def load_guidance_head(params):
    from diffusion.layers import (
        EdgeRegressionHead, NodeRegressionHead, GraphRegressionHead
    )
    if params['target'] in ['link', 'edge']:
        model = EdgeRegressionHead(**params)
    elif params['target'] == 'node':
        model = NodeRegressionHead(**params)
    elif params['target'] == 'graph':
        model = GraphRegressionHead(**params)
    else:
        raise NotImplementedError(f"Unsupported denoise model type {params['target']}")
    
    print('-'*100)
    print('Guidance head loaded!')
    print(model)
    return model

def load_diffusion_guidance_optim(config, device, ckpt_path=None, freeze_model=False):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        ckpt_config = ckpt['model_config']
        config = update(config, ckpt_config)

    model_params, diffusion_params, training_params = load_diffusion_params(config)
    model_params.num_timesteps = diffusion_params.num_timesteps
    denoise_model = load_denoise_model(model_params)

    guidance_params = config.guidance
    # TODO: input_dim from denoise_model
    guidance_params.input_dim = model_params.hidden_channels
    guidance_head = load_guidance_head(guidance_params)

    model = load_diffusion_guidance(diffusion_params, denoise_model, guidance_head, ckpt_path)
    num_params = sum(param.numel() for param in model.parameters())
    print(f'Num of parameters: {(num_params / 1e6):.4f}M')

    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
            # model = torch.nn.parallel.DistributedDataParallel(model)
        model = model.to(f'cuda:{device[0]}')
        model.device = f'cuda:{device[0]}'
    else:
        model = model.to(device)
        model.device = device
    
    if not freeze_model:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(guidance_head.parameters()), 
                                     lr=training_params.lr, weight_decay=training_params.weight_decay)
    else:
        optimizer = torch.optim.Adam(guidance_head.parameters(), lr=training_params.lr, 
                                     weight_decay=training_params.weight_decay)
    scheduler = None
    if training_params.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training_params.lr_decay)
    
    return config, model, optimizer, scheduler

# ----------------------------------------
# -------- downstream loaders --------
# ----------------------------------------
def load_downstream_model(params):
    # params = config.model
    assert params['target'] is not None
    if params['target'] == 'GNN':
        return GNN(**params)
    elif params['target'] == 'GCNWithPooling':
        return GCNWithPooling(**params)
    elif params['target'] == 'SimpleGCN':
        return SimpleGCN(**params)
    elif params['target'] == 'GIN':
        return GIN(**params)
    elif params['target'] == 'NCNGCN':
        return NCNGCN(**params)
    elif params['target'] == 'GINVirtualnode':
        return GINVirtualnode(**params)
    # elif params['target'] == 'GNNWithPooling':
    #     return GNNWithPooling

def load_predictor(params):
    assert params['target'] is not None
    if params['target'] == 'ncn':
        return CNLinkPredictor(**params)
    elif params['target'] == 'ncnc':
        return IncompleteCN1Predictor(**params)
    elif params['target'] == 'mlp':
        return MLPLinkPredictor(**params)
    elif params['target'] == 'cf':
        return CFLinkPredictor(**params)

def load_optimizer_and_scheduler(model_parameters, params):
    optimizer = torch.optim.Adam(model_parameters, lr=params.lr, 
                                 weight_decay=params.weight_decay)
    scheduler = None
    if params.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.lr_decay)
    
    return optimizer, scheduler

# ----------------------------------------
# -------- GDSS loaders --------
# ----------------------------------------
def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model

def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler

def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b

def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde

def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps)
    return loss_fn

def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num  # temporary fix

    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth, 
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final, 
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv':config_m.conv}
    else:
        params_x = {'model_type':config_m.x, 'max_feat_num':max_feat_num, 'depth':config_m.depth, 'nhid':config_m.nhid}
    params_adj = {'model_type':config_m.adj, 'max_feat_num':max_feat_num, 'max_node_num':config.data.max_node_num, 
                    'nhid':config_m.nhid, 'num_layers':config_m.num_layers, 'num_linears':config_m.num_linears, 
                    'c_init':config_m.c_init, 'c_hid':config_m.c_hid, 'c_final':config_m.c_final, 
                    'adim':config_m.adim, 'num_heads':config_m.num_heads, 'conv':config_m.conv}
    return params_x, params_adj

def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
    path = f'./checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    ckpt_dict= {'config': ckpt['model_config'], 'params_x': ckpt['params_x'], 'x_state_dict': ckpt['x_state_dict'],
                'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt['adj_state_dict']}
    if config.sample.use_ema:
        ckpt_dict['ema_x'] = ckpt['ema_x']
        ckpt_dict['ema_adj'] = ckpt['ema_adj']
    if return_ckpt:
        ckpt_dict['ckpt'] = ckpt
    return ckpt_dict

def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DDP models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model

def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels