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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=1000)  # sota 1000
    parser.add_argument('--patience', type=int, default=50)  # sota 80

    parser.add_argument('--learning_rate', type=float, default=1e-4)
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
                        default='CubeMLP', help='one of {MultimodalTransformer, Bi_LSTM, CubeMLP}')
    # Scale standards
    parser.add_argument('--center_score', type=parse_list, default=[0, 11, 17, 24, 32])

    parser.add_argument('--test_mae_history_path', type=str, default='./test_mae_history_standards.txt')


    # Data
    parser.add_argument('--data', type=str, default='/home/liu70kg/PycharmProjects/SEARCH_process-master/SEARCH_data_')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]


    # kwargs.best_model_Configuration_Log = "./Experiment_4_3_2.txt"
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


# 定义目标函数
def objective(trial):
    reset_seed1 = trial.suggest_int("reset_seed1", 1, 1000)
    reset_seed(reset_seed1)  # 必不可少,否则复现不出来
    # 定义 6 个超参数
    batch_size = trial.suggest_categorical("batch_size", [128, 64, 32, 256])  # 整数范围
    activation = trial.suggest_categorical("activation", ["leakyrelu", "prelu", "relu", "rrelu", "tanh"])  # 离散集合
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"

    # Setting the config for each stage
    train_config = get_config(mode='train')
    train_config.batch_size = batch_size
    train_config.activation = activation_dict[activation]

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
    MAE = solver.train()

    return MAE  # 返回目标值


if __name__ == '__main__':

    # Setting random seed
    random_name = str(random())
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    study_name = "optimization_study"
    storage = f"sqlite:///{study_name}.db"
    # 开始超参数优化
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)
    # 如果已经完成优化，直接加载，无需重新优化
    if len(study.trials) == 0:
        # 定义固定参数
        fixed_params = {
            "batch_size": 256,
            "activation": "relu",

        }

        # 手动插入固定参数为一个试验
        study.enqueue_trial(fixed_params)

        study.optimize(objective, n_trials=200)

        # 打印最佳参数
        print("Best parameters:", study.best_params)
        print("Best MAE:", study.best_value)

        # 可视化优化结果
        vis.plot_optimization_history(study).show()
        vis.plot_parallel_coordinate(study).show()
    else:
        print("Loaded existing Study from database.")

        best_trial = study.best_trial
        print("Best trial value from Optuna:", best_trial.value)  # 应该是 5.977

        # 用 Optuna 的方式执行一次试试
        mae1 = objective(optuna.trial.FixedTrial(best_trial.params))
        print(f"MAE by objective(): {mae1}")

        # 加载 Study
        loaded_study = optuna.load_study(study_name="optimization_study", storage=f"sqlite:///{study_name}.db")

        # 自定义并行坐标图的线条粗细
        fig = plot_parallel_coordinate(loaded_study)

        # 修改线条宽度
        for trace in fig.data:
            if trace.type == 'scatter':
                trace.line.width = 5  # 修改线条宽度为 2

        # 显示图
        fig.show()


# 展示的命令
# pip install optuna-dashboard
# optuna-dashboard sqlite:///optimization_study.db