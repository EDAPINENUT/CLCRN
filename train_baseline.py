from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from supervisor import Supervisor
import yaml
import numpy as np
from pathlib import Path
import os
import json


def main(args):

    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        path = Path(supervisor_config['train']['log_dir'])/supervisor_config['train']['experiment_name']
        path.mkdir(exist_ok = True, parents = True)

        sv_param = os.path.join(path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(supervisor_config, file_obj)
        supervisor = Supervisor(**supervisor_config)

        supervisor.train()
        supervisor._test_final_n_epoch(1)


def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024, type=int, #2021~2025
                        help='seed for reproducity')
    parser.add_argument('--config_filename', default='./experiments/config_baseline.yaml', type=str,
                        help='Configuration filename for restoring the model.')
                        
    args = parser.parse_args()
    SetSeed(args.seed)          

    main(args)
