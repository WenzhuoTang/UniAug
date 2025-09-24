import yaml
# from easydict import EasyDict as edict

from omegaconf import OmegaConf


def get_config(config, seed):
    # config_dir = f'./config/{config}.yaml'
    # config = edict(yaml.load(open(config, 'r'), Loader=yaml.SafeLoader))
    config = OmegaConf.load(config)
    config.train.seed = seed

    return config

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config