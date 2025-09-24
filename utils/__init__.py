import os
import torch
import random
import numpy as np

from .misc import *
from .augment import *
from .solver import load_generator

def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_seed(rndseed, cuda: bool = True, use_dgl: bool = False, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if use_dgl:
        import dgl
        dgl.seed(rndseed)
        dgl.random.seed(rndseed)
    print(f"Setting global random seed to {rndseed}")

def dict_of_dicts_to_dict(input_dict):
    output_dict = {}
    for i in input_dict.keys():
        for j in input_dict[i].keys():
            output_dict[f'{i}_{j}'] = input_dict[i][j]
    return output_dict

def flatten_dict(dictionary, stop_keys=None, return_flag=True):
    ''' 
    Inplace flatten nested dictionary. 
    Will stop once met the stop_keys.
    '''
    for k, v in list(dictionary.items()):
        if type(v) == dict:
            if not (stop_keys is not None and any([x in stop_keys for x in v.keys()])):
                flatten_dict(v, stop_keys, return_flag=False)
                dictionary.pop(k)
                for k2, v2 in v.items():
                    dictionary[k+"."+k2] = v2
    if return_flag:
        return dictionary

def unflatten_dict(dictionary):
    ''' Unflatten nested dictionary. '''
    out_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = out_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value

    return out_dict