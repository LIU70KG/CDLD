import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from utils import to_gpu
from utils import ReverseLayerF
from torch.nn.functional import pairwise_distance

def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# 定义单个CubeMLP块
class CubeMLPBlock(nn.Module):
    def __init__(self, seq_len, out_seq_len, feature_dim, out_feat_dim, num_modalities):
        super().__init__()
        self.seq_len = seq_len
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        self.mlp_seq = nn.Sequential(
            nn.Linear(seq_len, out_seq_len),
            nn.ReLU(),
            nn.Linear(out_seq_len, out_seq_len)
        )
        self.mlp_modality = nn.Sequential(
            nn.Linear(num_modalities, num_modalities),
            nn.ReLU(),
            nn.Linear(num_modalities, num_modalities)
        )
        self.mlp_channel = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_feat_dim)
        )

        # 定义层归一化
        self.norm_seq = nn.LayerNorm([out_seq_len, num_modalities, feature_dim])
        self.norm_modality = nn.LayerNorm([out_seq_len, num_modalities, feature_dim])
        self.norm_channel = nn.LayerNorm([out_seq_len, num_modalities, out_feat_dim])

        # 定义残差连接的线性变换
        self.residual_seq = nn.Linear(seq_len, out_seq_len)
        self.residual_channel = nn.Linear(feature_dim, out_feat_dim)

    def forward(self, x):
        # 序列混合 (MLP-L)
        x_seq = self.mlp_seq(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_res_seq = self.residual_seq(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x_seq + self.norm_seq(x_res_seq)

        # 模态混合 (MLP-M)
        x_modality = self.mlp_modality(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x + self.norm_modality(x_modality)

        # 通道混合 (MLP-D)
        x_channel = self.mlp_channel(x)
        x_res_channel = self.residual_channel(x)
        x = x_channel + self.norm_channel(x_res_channel)

        return x



class CubeMLP(nn.Module):
    def __init__(self, config, seq_len=20, num_modalities=3, hidden_dim=128, last_seq=3, last_cannel=10):
        super().__init__()
        self.seq_len = seq_len
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        feature_dims = [config.txt_size, config.visual_size, config.acoustic_size]
        # 预处理：将各模态映射到统一维度

        self.modal_projs_t = nn.Linear(feature_dims[0], hidden_dim)
        self.modal_projs_v = nn.Linear(feature_dims[1], hidden_dim)
        self.modal_projs_a = nn.Linear(feature_dims[2], hidden_dim)

        # 3层 CubeMLP block
        self.block1 = CubeMLPBlock(20, seq_len, 128, 64, num_modalities)
        self.block2 = CubeMLPBlock(seq_len, 10, 64, 32, num_modalities)
        self.block3 = CubeMLPBlock(10, last_seq, 32, last_cannel, num_modalities)

        # (self, seq_len, out_seq_len, feature_dim, out_feat_dim, num_modalities)

        # 预测头
        self.predictor = nn.Linear(last_seq * num_modalities * last_cannel, 1)

    def alignment(self, text, visual, acoustic, lengths, label_area):
        text = text.permute(1, 0, 2)
        text = self.modal_projs_t(text)

        visual = visual.permute(1, 0, 2)
        visual = self.modal_projs_v(visual)

        acoustic = acoustic.permute(1, 0, 2)
        acoustic = self.modal_projs_a(acoustic)
        x = torch.stack([text, visual, acoustic], dim=2)

        x = self.block1(x)  # [batch, 20, 2, 64]
        x = self.block2(x)  # [batch, 10, 2, 32]
        x = self.block3(x)  # [batch, 3, 2, 10]

        x_flat = x.view(x.size(0), -1)
        out = self.predictor(x_flat)
        return out, x

    def forward(self, text, video, acoustic, lengths, label_area):
            o, h = self.alignment(text, video, acoustic, lengths, label_area)
            return o, h
