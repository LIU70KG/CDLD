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
from optuna.visualization import plot_parallel_coordinate
import torch
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import optuna
import shap  # 确保已安装 SHAP：pip install shap
import optuna.visualization as vis
# DASS-21 : 抑郁量表≤9分为正常，l0～l3分为轻度，14～20分为中度，21～27分为重度，≥28分为非常严重；

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

        self.dataset_dir = self.data
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


# 自定义一个函数来解析字符串并返回列表
def parse_list(value):
    # 去除空格，并将字符串按逗号分割，然后转换成整数列表
    return [int(x) for x in value.split(',')]


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_Configuration_Log', type=str, default='./best_model_Configuration_Log.txt',
                        help='Load the best model to save features')
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=1000)  # sota 1000
    parser.add_argument('--patience', type=int, default=300)  # sota 80

    parser.add_argument('--diff_weight', type=float, default=0.3)  # MISA 超参数
    parser.add_argument('--sim_weight', type=float, default=0.5)   # MISA 超参数
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)

    parser.add_argument('--class_weight', type=float, default=15.0)  # 15.0 3.0
    parser.add_argument('--shifting_weight', type=float, default=3.0)  # 3.0 2.0
    parser.add_argument('--order_center_weight', type=float, default=1.0)
    parser.add_argument('--ce_loss_weight', type=float, default=1.0)
    parser.add_argument('--pred_center_score_weight', type=float, default=0.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)  # 1e-4
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')
    # Model
    parser.add_argument('--model', type=str,
                        default='MISA_CMDC', help='one of {Simple_Fusion_Network, MISA_CMDC}')
    # Scale standards 5
    # parser.add_argument('--interval_Scale', type=str, default='standards')
    # parser.add_argument('--interval_num', type=int, default=5)
    # parser.add_argument('--center_score', type=list, default=[7, 16, 24, 33, 41])
    # parser.add_argument('--min_shift', type=float, default=-7)
    # parser.add_argument('--max_shift', type=float, default=6)
    # parser.add_argument('--test_mae_history_path', type=str, default='./test_mae_history_standards.txt')

    #  # Scale average 5
    # parser.add_argument('--interval_Scale', type=str, default='average')
    # parser.add_argument('--interval_num', type=int, default=5)
    # parser.add_argument('--center_score', type=list, default=[4, 13, 22, 31, 40])
    # parser.add_argument('--min_shift', type=float, default=-4.0)
    # parser.add_argument('--max_shift', type=float, default=5.0)
    # parser.add_argument('--test_mae_history_path', type=str, default='./test_mae_history_average.txt')


    # # Scale standards 6
    parser.add_argument('--interval_Scale', type=str, default='standards')
    parser.add_argument('--interval_num', type=int, default=6)
    parser.add_argument('--center_score', type=list, default=[3, 10, 16, 24, 33, 41])
    parser.add_argument('--min_shift', type=float, default=-4)
    parser.add_argument('--max_shift', type=float, default=4)
    parser.add_argument('--test_mae_history_path', type=str, default='./test_mae_history_standards.txt')


    # # Scale average 6
    # parser.add_argument('--interval_Scale', type=str, default='average')
    # parser.add_argument('--interval_num', type=int, default=6)
    # parser.add_argument('--center_score', type=list, default=[3, 10, 17, 25, 33, 41])
    # parser.add_argument('--min_shift', type=float, default=-3.0)
    # parser.add_argument('--max_shift', type=float, default=4.0)
    # parser.add_argument('--test_mae_history_path', type=str, default='./test_mae_history_average.txt')

    # Data
    # parser.add_argument('--data', type=str,
    #                     default='/home/liu70kg/PycharmProjects/AVEC_process-master/AVEC2014_Northwind/AVEC2014_fea_')  # 150个样本，仅仅包含Northwind
    # parser.add_argument('--data', type=str,
    #                     default='/home/liu70kg/PycharmProjects/AVEC_process-master/AVEC2014_Freeform/AVEC2014_fea_')  # 150个样本，仅仅包含Freeform

    parser.add_argument('--data', type=str,
                        default='/home/liu70kg/PycharmProjects/AVEC_process-master/AVEC2014_con_Freeform_Northwind_fa/fea_')  # 150个样本，Northwind拼接Freeform

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    if kwargs.model == "Simple_Fusion_Network":
        kwargs.best_model_Configuration_Log = "./Simple_Fusion_Network_report.txt"
    elif kwargs.model == "MISA_CMDC":
        kwargs.best_model_Configuration_Log = "./MISA_CMDC_backbone_report.txt"

    # kwargs.best_model_Configuration_Log = "./Experiment_4_3_2.txt"
    # print(kwargs.data)
    # print(kwargs.model)
    # print(kwargs.best_model_Configuration_Log)

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)



if __name__ == '__main__':

    # Setting random seed
    random_name = str(random())
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Setting the config for each stage
    train_config = get_config(mode='train')
    test_config = get_config(mode='test')
    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle=True)
    test_data_loader = get_loader(test_config, shuffle=False)

    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, test_config, train_data_loader, test_data_loader,is_train=True)


    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()
