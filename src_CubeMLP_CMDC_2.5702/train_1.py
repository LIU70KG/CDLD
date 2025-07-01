# 作者：刘成广
# 时间：2024/7/17 下午2:23
# 作者：刘成广
# 时间：2024/7/16 下午10:09
import os
import pickle
import numpy as np
from random import random
from data_loader import get_loader
from solver import Solver

import torch
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '../glove.840B.300d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY'), 'daic-woz': data_dir.joinpath('DAIC-WOZ'), 'cmdc': data_dir.joinpath('CMDC')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str



def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_Configuration_Log', type=str, default='./CubeMLP_backbone_report.txt',
                        help='Load the best model to save features')
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=500)  # sota 500
    parser.add_argument('--patience', type=int, default=100)  # sota 100

    parser.add_argument('--diff_weight', type=float, default=0.3)
    parser.add_argument('--sim_weight', type=float, default=1.0)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)

    parser.add_argument('--class_weight', type=float, default=5.0)
    parser.add_argument('--shifting_weight', type=float, default=2.0)
    parser.add_argument('--order_center_weight', type=float, default=1.5)
    parser.add_argument('--ce_loss_weight', type=float, default=3.0)
    parser.add_argument('--pred_center_score_weight', type=float, default=0.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='prelu')
    # Model
    parser.add_argument('--model', type=str,
                        default='CubeMLP', help='one of {Simple_Fusion_Network, MISA_CMDC, CubeMLP}')

    # Data
    parser.add_argument('--data', type=str, default='cmdc')  # cmdc DAIC-WOZ--------------------------------------
    parser.add_argument('--center_score', type=list, default=[2, 7, 12, 17, 22])
    # cmdc的5折交叉验证
    parser.add_argument("--cross_validation", type=str,
                        choices=["cmdc_data_all_modal_1", "cmdc_data_all_modal_2", "cmdc_data_all_modal_3",
                                 "cmdc_data_all_modal_4", "cmdc_data_all_modal_5"], default="cmdc_data_all_modal_1")
    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)
    print(kwargs.model)
    print(kwargs.best_model_Configuration_Log)

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


def reset_seed(seed=336):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 获取当前工作区目录
    current_directory = os.getcwd()
    print("当前工作区:", current_directory)
    # Setting random seed
    random_name = str(random())
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    reset_seed(188)
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # print(train_config)
    if train_config.data == 'cmdc' or train_config.data == 'CMDC':
        # Creating pytorch dataloaders
        train_data_loader = get_loader(train_config, shuffle=True)
        test_data_loader = get_loader(test_config, shuffle=False)
        dev_data_loader = test_data_loader

        # Solver is a wrapper for model traiing and testing
        solver = Solver
        solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader,
                        is_train=True)

        # Build the model
        solver.build()

        # Train the model (test scores will be returned based on dev performance)
        solver.train()



    else:
        # Creating pytorch dataloaders
        train_data_loader = get_loader(train_config, shuffle=True)
        dev_data_loader = get_loader(dev_config, shuffle=False)
        test_data_loader = get_loader(test_config, shuffle=False)

        # Solver is a wrapper for model traiing and testing
        solver = Solver
        solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader,
                        is_train=True)

        # Build the model
        solver.build()

        # Train the model (test scores will be returned based on dev performance)
        solver.train()
