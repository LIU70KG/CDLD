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
    parser.add_argument('--activation', type=str, default='relu')
    # Model
    parser.add_argument('--model', type=str,
                        default='CubeMLP', help='one of {Simple_Fusion_Network, MISA_CMDC, CubeMLP}')

    # Data
    parser.add_argument('--data', type=str, default='DAIC-WOZ')  # cmdc DAIC-WOZ--------------------------------------
    parser.add_argument('--center_score', type=list, default=[2, 7, 12, 17, 22])

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


# 定义目标函数
def objective(trial):
    reset_seed1 = trial.suggest_int("reset_seed1", 1, 1000)
    reset_seed(reset_seed1)  # 必不可少,否则复现不出来
    # 定义 6 个超参数
    # batch_size = trial.suggest_categorical("batch_size", [128, 64, 32, 256])  # 整数范围
    activation = trial.suggest_categorical("activation", ["leakyrelu", "prelu", "relu", "rrelu", "tanh"])  # 离散集合
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"

    # Setting the config for each stage
    train_config = get_config(mode='train')
    # train_config.batch_size = batch_size
    train_config.activation = activation_dict[activation]

    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

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
            # "batch_size": 256,
            "activation": "relu",

        }

        # 手动插入固定参数为一个试验
        study.enqueue_trial(fixed_params)

        study.optimize(objective, n_trials=30)

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
        print("Best trial params from Optuna:", best_trial.params)  # 应该是 5.977
        # # 用 Optuna 的方式执行一次试试
        # mae1 = objective(optuna.trial.FixedTrial(best_trial.params))
        # print(f"MAE by objective(): {mae1}")

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