import os
from os.path import join, dirname
import torch
import math

### PATH CONFIGS ###

_root_directory = dirname(__file__)

path_configs = {'root_dir': _root_directory,
                'world_data_dir': join(_root_directory, 'world_data'),
                'model_data_dir': join(_root_directory, 'model_data')
                }


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# safe_mkdir(path_configs['world_data_dir'])
# safe_mkdir(path_configs['model_data_dir'])
# safe_mkdir(join(path_configs['world_data_dir'], 'images'))

### HARDWARE CONFIGS ###

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    _device = 0
else:
    _device = 'cpu'

# _device = 'cpu'  # TODO: debug CUDA
machine_configs = {'device': _device}


def move(x):
    return x.to(_device)

# hyperparameters
# For GANzoo, use sigma=0.01, alpha=1.1
hp = {'sigma': 0.02, # Greater sigma = greater overall covariance
      'alpha': 1.1, # Smaller alpha = distance has a larger effect
      'lscale': 1.6, # Affects initial megatile; greater lscale = more similar tiles
      'model_family': 'poincare'
      }
