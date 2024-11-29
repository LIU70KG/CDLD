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


# let's define a simple model that can deal with multimodal variable length sequence
class MISA_CMDC(nn.Module):
    def __init__(self, config):
        super(MISA_CMDC, self).__init__()

        self.config = config
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        self.num_classes = self.config.interval_num

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.vrnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.arnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        self.private_v.add_module('private_v_activation_1', self.activation)
        self.private_v.add_module('private_v_activation_1_norm', nn.LayerNorm(config.hidden_size))

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        self.private_a.add_module('private_a_activation_3', self.activation)
        self.private_a.add_module('private_a_activation_1_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.shared.add_module('shared_activation_1', nn.Sigmoid())
        self.shared.add_module('shared_activation_1', self.activation)
        self.shared.add_module('shared_1_norm', nn.LayerNorm(config.hidden_size))
        ##########################################
        # reconstruct
        ##########################################
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))
        ####################################################################################
        # ----------将各样本特征映射到类特征中----------------
        self.Dimensionality_reduction = nn.Linear(in_features=self.config.hidden_size * 4, out_features=32)
        self.features_to_center = nn.Linear(in_features=32, out_features=32)
        # self.features_to_center = nn.Sequential()
        # self.features_to_center.add_module('center',
        #                           nn.Linear(in_features=64, out_features=64))
        # # self.features_to_center.add_module('center_activation', nn.Sigmoid())
        # self.features_to_center.add_module('center_activation', self.activation)
        # self.features_to_center.add_module('center_layer_norm', nn.LayerNorm(config.hidden_size))
        # -----------将类特征去预测类别--------------------
        self.fc_class = nn.Sequential()
        # self.fc_class.add_module('fc1', nn.Linear(in_features=self.config.hidden_size,
        #                                                    out_features=10))
        self.fc_class.add_module('fc2', nn.Linear(in_features=32, out_features=self.num_classes))
        # self.fc_class.add_module('softmax', nn.Softmax(dim=1))

        # ----------将各样本特征提取回归特征----------------
        # self.features_to_sample = nn.Sequential()
        # self.features_to_sample.add_module('sample',
        #                           nn.Linear(in_features=self.config.hidden_size * 6, out_features=config.hidden_size))
        # self.features_to_sample.add_module('sample_activation', nn.Sigmoid())
        # -----------特征去回归类别--------------------
        self.fc_score = nn.Linear(in_features=32, out_features= self.output_size)

        # ----------预测各样本的不确定性----------------   不知道为什么，将算后面不使用这个语句，加或者不加，或者加在self.fc_class 的前后，都对最后有影响
        # self.alpha = nn.Sequential()
        # self.alpha.add_module('alpha', nn.Linear(in_features=self.config.hidden_size, out_features=1))
        # self.alpha.add_module('alpha_activation', nn.Sigmoid())
        self.alpha = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())
        self.Softmax = nn.Softmax(dim=1)
        # # ----------初始化类别中心----------------
        self.centers = nn.Parameter(torch.zeros(self.config.interval_num, 32))  # 设置全是0？
        self.fc_center_score = nn.Linear(in_features=32, out_features=self.output_size)
        # for i in range(5):
        #     self.centers.data[i] = i * torch.ones(config.hidden_size)

        # ----------预测偏移量[-2, 2]-------------------
        # self.fc_shift = nn.Sequential()
        # self.fc_shift.add_module('shift1', nn.Linear(in_features=self.config.hidden_size * 7,
        #                                                    out_features=self.config.hidden_size))
        # # self.fc_shift.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fc_shift.add_module('fusion_layer_1_activation', self.activation)
        # self.fc_shift.add_module('shift2', nn.Linear(in_features=self.config.hidden_size, out_features=output_size))
        # ----------预测偏移量[-2, 2]-------------------
        # self.fc_shift = nn.Linear(config.hidden_size, output_size)
        self.fc_shift = nn.Linear(32*2, output_size)
        # self.fc_shift = nn.Linear(self.num_classes, output_size)
        # ----------直接回归分数-------------------
        # self.fusion = nn.Sequential()
        # self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        # self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
        ####################################################################################

        self.vlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))

        # 原来是，但会出现警告
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, visual, acoustic, lengths, label_area, mode):

        batch_size = lengths.size(0)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_v + self.utt_shared_a) / 2.0)

        # For reconstruction
        self.reconstruct()

        # 方法1,MISA的4个特征
        h = torch.stack((self.utt_private_v, self.utt_private_a,
                         self.utt_shared_v, self.utt_shared_a), dim=0)
        # h = torch.stack((self.utt_t_orig, self.utt_private_v, self.utt_a_orig,), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        h = self.Dimensionality_reduction(h)
        features_class = self.features_to_center(h)
        # features_sample = self.features_to_sample(h)
        # pred = self.fc_score(features_sample)
        # 欧式距离-----------------------------
        # 对特征进行规范化处理
        if mode == "train":
            batch_size, feature_dim = features_class.shape
            label_area = label_area.view(-1).long()
            # 计算每个类的平均特征作为类中心
            centers = []
            for i in range(self.num_classes):
                mask = (label_area == i)
                if mask.sum() == 0:
                    # 防止某类没有样本，避免除以零错误
                    centers.append(torch.zeros(feature_dim).cuda())
                elif mask.sum() == 1:
                    centers.append(features_class[mask].mean(dim=0) * 0.9 + torch.randn(
                        feature_dim).cuda() * 0.1)  # 添加一些噪声以提高数值稳定性
                else:
                    centers.append(features_class[mask].mean(dim=0))
            centers = torch.stack(centers)
            # 将新的中心点转换为nn.Parameter并更新
            # self.centers = nn.Parameter(centers)
            # self.centers.data = centers
            alpha = 0.1
            if torch.all(self.centers.data.eq(0)):
                self.centers.data = centers  # 使用 .data 方式进行增量更新
            else:
                self.centers.data = (1 - alpha) * self.centers.data + alpha * centers  # 使用 .data 方式进行增量更新

            # 计算中心损失：让每个样本离当前类中心点尽量近，而离其他类中心点尽量远
            with torch.no_grad():
                mask = torch.eye(self.num_classes).cuda()[label_area]
                target = label_area.view(-1, 1)
                class_range = torch.arange(0, self.num_classes).view(1, -1).cuda()
                weight = torch.abs(class_range - target).float()

            # 计算每个样本与各类中心的欧几里得距离
            dists = torch.cdist(features_class, centers, p=2)
            # center_loss = (dists * mask).sum() / batch_size  # 当前类中心点的距离损失
            # center_loss -= (dists * weight).sum() / (batch_size * (self.num_classes - 1))
            # 使用反比例函数将距离转换为相似度
            sim = 1 / (1 + dists + 1e-8)
            # # 计算相似度损失1
            # positive_similarity = (sim * mask).sum() / batch_size  # 当前类中心点的相似度
            # negative_similarity = (sim * weight).sum() / (batch_size * (self.num_classes - 1))# 其他类中心点的相似度
            # # negative_similarity = (cos_sim * (1 - mask)).sum() / (batch_size * (self.num_classes - 1))
            # 计算相似度损失2

            positive_similarity = (sim * mask).sum() / batch_size  # 当前类中心点的相似度
            negative_similarity = (sim * weight).sum() / (batch_size * (self.num_classes - 1))# 其他类中心点的相似度
            # negative_similarity = (cos_sim * (1 - mask)).sum() / (batch_size * (self.num_classes - 1))
            # 确保相似度损失合理
            order_center_loss = 1 - positive_similarity + negative_similarity

            # # 有序性约束损失
            # order_loss = 0
            # for i in range(5 - 1):
            #     for j in range(i + 1, 5):
            #         order_loss += torch.relu((j - i) - torch.norm(centers[j] - centers[i]))
            # order_center_loss = center_loss + order_loss
            # uncertainty = self.alpha(features_class)
            # uncertainty = 1-U
            # uncertainty = 0
            uncertainty = torch.sum(sim * mask, dim=1)
            # p_class = (1-uncertainty) * self.fc_class(features_class)
            p_class = self.fc_class(features_class)
            pred = self.fc_score(features_class)
            pred_center_score = self.fc_center_score(self.centers.data)
            # --------------获取类似的区间中心特征---------------------
            # batch_size, feature_dim = features_sample.shape
            # label_area = label_area.view(-1).long()
            # # 计算每个类的平均特征作为类中心
            # centers = []
            # for i in range(self.num_classes):
            #     mask = (label_area == i)
            #     if mask.sum() == 0:
            #         # 防止某类没有样本，避免除以零错误
            #         centers.append(torch.zeros(feature_dim).cuda())
            #     else:
            #         centers.append(features_sample[mask].mean(dim=0))
            # centers = torch.stack(centers)
            # # 将新的中心点转换为nn.Parameter并更新
            # self.centers = nn.Parameter(centers)
            # -----------------------------------
            # features_shift = torch.cat((h, features_class), dim=1)
            # features_shift = h - torch.cat([features_class] * 6, dim=1)
            centers_batch = centers[label_area]  # (batch_size, feat_dim)
            # features_shift = features_class - centers_batch
            features_shift = torch.cat((h, centers_batch), dim=1)
            p_shift = self.fc_shift(features_shift)
            # p_shift = self.fc_shift(sim)
            p_shift = torch.clamp(p_shift, min=self.config.min_shift, max=self.config.max_shift)  # 输出限制

            # 预测
            # pred = self.fusion(h)
            # pred = 0
            # 方法2，3个modal特征直接拼接
            # h = torch.stack((self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=0)
            # h = self.transformer_encoder(h)
            # h = torch.cat((h[0], h[1], h[2]), dim=1)
            # features_class = self.features_to_center(h)
            #
            # uncertainty = self.alpha(features_class)
            # p_class = self.fc_class(features_class)
            # features_shift = torch.cat((h, features_class), dim=1)
            # # features_shift = h - torch.cat([features_class] * 6, dim=1)
            # p_shift = self.fc_shift(features_shift)
            # p_shift = torch.clamp(p_shift, min=-2.0, max=2.0)  # 输出限制在[−2,2] 之间

            return pred, pred_center_score, p_class, p_shift, uncertainty, order_center_loss
        if mode == "test":
            p_class = self.fc_class(features_class)

            # features_sample = self.features_to_sample(h)
            # pred = self.fc_score(features_sample)
            pred = self.fc_score(features_class)
            pred_center_score = self.fc_center_score(self.centers.data)
            label_area = label_area.view(-1).long()
            centers_batch = self.centers[label_area]  # (batch_size, feat_dim)
            # features_shift = features_class - centers_batch
            features_shift = torch.cat((h, centers_batch), dim=1)
            p_shift = self.fc_shift(features_shift)

            # dists = torch.cdist(features_class, self.centers, p=2)
            # sim = 1 / (1 + dists)
            # p_shift = self.fc_shift(sim)
            p_shift = torch.clamp(p_shift, min=self.config.min_shift, max=self.config.max_shift)  # 输出限制

            # 预测
            # pred = self.fusion(h)
            return pred, pred_center_score, p_class, p_shift

        if mode == "tsne":
            return features_class

    def reconstruct(self, ):

        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_v, utterance_a):

        # Projecting to same sized space
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, video, acoustic, lengths, label_area, mode):
        if mode == "train":
            pred, pred_center_score, p_class, p_shifting, uncertainty, order_center_loss = self.alignment(video, acoustic, lengths, label_area, mode)
            return pred, pred_center_score, p_class, p_shifting, uncertainty, order_center_loss

        if mode == "test":
            pred, pred_center_score, p_class, p_shifting = self.alignment(video, acoustic, lengths, label_area, mode)
            return pred, pred_center_score, p_class, p_shifting

        if mode == "tsne":
            features_class = self.alignment(video, acoustic, lengths, label_area, mode)
            return features_class


# let's define a simple model that can deal with multimodal variable length sequence
class Simple_Fusion_Network(nn.Module):
    def __init__(self, config):
        super(Simple_Fusion_Network, self).__init__()

        self.config = config
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        self.num_classes = self.config.interval_num

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.vrnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.arnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.shared.add_module('shared_activation_1', nn.Sigmoid())
        self.shared.add_module('shared_activation_1', self.activation)
        self.shared.add_module('shared_1_norm', nn.LayerNorm(config.hidden_size))

        ####################################################################################
        # ----------将各样本特征映射到类特征中----------------
        # self.features_to_center = nn.Sequential()
        # self.features_to_center.add_module('center',
        #                           nn.Linear(in_features=self.config.hidden_size * 2, out_features=config.hidden_size))
        # # self.features_to_center.add_module('center_activation', nn.Sigmoid())
        # self.features_to_center.add_module('center_activation', self.activation)
        # self.features_to_center.add_module('center_layer_norm', nn.LayerNorm(config.hidden_size))

        self.Dimensionality_reduction = nn.Linear(in_features=self.config.hidden_size * 2, out_features=32)
        self.features_to_center = nn.Linear(in_features=32, out_features=32)
        # -----------将类特征去预测类别--------------------
        self.fc_class = nn.Sequential()
        # self.fc_class.add_module('fc1', nn.Linear(in_features=self.config.hidden_size,
        #                                                    out_features=10))
        self.fc_class.add_module('fc2', nn.Linear(in_features=32, out_features=self.num_classes))
        # self.fc_class = nn.Linear(in_features=10, out_features=self.num_classes)
        # self.fc_class.add_module('softmax', nn.Softmax(dim=1))

        # -----------特征去回归类别--------------------
        self.fc_score = nn.Sequential()
        # self.fc_score.add_module('fc11', nn.Linear(in_features=self.config.hidden_size,
        #                                                    out_features=10))
        self.fc_score.add_module('fc22', nn.Linear(in_features=32, out_features=self.output_size))

        self.fc_center_score = nn.Linear(in_features=32, out_features=self.output_size)
        # self.fc_score = nn.Linear(in_features=self.config.hidden_size, out_features= output_size)

        # ----------预测各样本的不确定性----------------   不知道为什么，将算后面不使用这个语句，加或者不加，或者加在self.fc_class 的前后，都对最后有影响
        # self.alpha = nn.Sequential()
        # self.alpha.add_module('alpha', nn.Linear(in_features=self.config.hidden_size, out_features=1))
        # self.alpha.add_module('alpha_activation', nn.Sigmoid())
        # self.alpha = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())
        self.alpha = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.Softmax = nn.Softmax(dim=1)
        # # ----------初始化类别中心----------------
        # self.centers = nn.Parameter(torch.randn(5, config.hidden_size))
        # self.fc_shift = nn.Linear(config.hidden_size*2, output_size)
        # self.centers = nn.Parameter(torch.randn(5, 32))
        self.centers = nn.Parameter(torch.zeros(self.config.interval_num, 32))  # 设置全是0？
        self.fc_shift = nn.Linear(32*2, output_size)
        ####################################################################################

        self.vlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))

        # 原来是，但会出现警告
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, visual, acoustic, lengths, label_area, mode):

        batch_size = lengths.size(0)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Projecting to same sized space
        self.utt_v_orig = self.project_v(utterance_video)
        self.utt_a_orig = self.project_a(utterance_audio)
        self.utt_shared_v = self.shared(self.utt_v_orig)
        self.utt_shared_a = self.shared(self.utt_a_orig)

        # 方法1,MISA的4个特征
        # h = torch.stack((self.utt_v_orig, self.utt_a_orig), dim=0)
        h = torch.stack((self.utt_shared_v, self.utt_shared_a), dim=0)
        # # h = torch.stack((self.utt_t_orig, self.utt_private_v, self.utt_a_orig,), dim=0)
        # h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1]), dim=1)
        h = self.Dimensionality_reduction(h)
        features_class = self.features_to_center(h)
        # -----------------------------------------------------------------

        # 欧式距离-----------------------------
        # 对特征进行规范化处理
        if mode == "train":
            batch_size, feature_dim = features_class.shape
            label_area = label_area.view(-1).long()
            # 计算每个类的平均特征作为类中心
            centers = []
            for i in range(self.num_classes):
                mask = (label_area == i)
                if mask.sum() == 0:
                    # 防止某类没有样本，避免除以零错误
                    centers.append(torch.zeros(feature_dim).cuda())
                elif mask.sum() == 1:
                    centers.append(features_class[mask].mean(dim=0) * 0.9 + torch.randn(
                        feature_dim).cuda() * 0.1)  # 添加一些噪声以提高数值稳定性
                else:
                    centers.append(features_class[mask].mean(dim=0))
            centers = torch.stack(centers)
            # 将新的中心点转换为nn.Parameter并更新
            # self.centers = nn.Parameter(centers)
            # self.centers.data = centers
            alpha = 0.1
            if torch.all(self.centers.data.eq(0)):
                self.centers.data = centers  # 使用 .data 方式进行增量更新
            else:
                self.centers.data = (1 - alpha) * self.centers.data + alpha * centers  # 使用 .data 方式进行增量更新

            # 计算中心损失：让每个样本离当前类中心点尽量近，而离其他类中心点尽量远
            with torch.no_grad():
                mask = torch.eye(self.num_classes).cuda()[label_area]
                target = label_area.view(-1, 1)
                class_range = torch.arange(0, self.num_classes).view(1, -1).cuda()
                weight = torch.abs(class_range - target).float()

            # 计算每个样本与各类中心的欧几里得距离
            dists = torch.cdist(features_class, centers, p=2)
            # center_loss = (dists * mask).sum() / batch_size  # 当前类中心点的距离损失
            # center_loss -= (dists * weight).sum() / (batch_size * (self.num_classes - 1))
            # 使用反比例函数将距离转换为相似度
            sim = 1 / (1 + dists + 1e-8)
            # 计算相似度损失

            positive_similarity = (sim * mask).sum() / batch_size  # 当前类中心点的相似度
            negative_similarity = (sim * weight).sum() / (batch_size * (self.num_classes - 1))# 其他类中心点的相似度
            # 确保相似度损失合理
            order_center_loss = 1 - positive_similarity + negative_similarity

            uncertainty = torch.sum(sim * mask, dim=1)
            # p_class = (1-uncertainty) * self.fc_class(features_class)
            p_class = self.fc_class(features_class)
            pred = self.fc_score(h)
            pred_center_score = self.fc_center_score(self.centers.data)

            # -----------------------------------
            # features_shift = torch.cat((h, features_class), dim=1)
            # features_shift = h - torch.cat([features_class] * 6, dim=1)
            centers_batch = centers[label_area]  # (batch_size, feat_dim)
            # features_shift = features_class - centers_batch
            # features_shift = torch.cat((features_class, centers_batch), dim=1)
            features_shift = torch.cat((h, centers_batch), dim=1)
            p_shift = self.fc_shift(features_shift)
            # p_shift = self.fc_shift(sim)
            p_shift = torch.clamp(p_shift, min=self.config.min_shift, max=self.config.max_shift)  # 输出限制

            return pred, pred_center_score, p_class, p_shift, uncertainty, order_center_loss
        if mode == "test":
            p_class = self.fc_class(features_class)

            # features_sample = self.features_to_sample(h)
            # pred = self.fc_score(features_sample)
            pred = self.fc_score(h)
            pred_center_score = self.fc_center_score(self.centers.data)
            label_area = label_area.view(-1).long()
            centers_batch = self.centers[label_area]  # (batch_size, feat_dim)
            # features_shift = features_class - centers_batch
            # features_shift = torch.cat((features_class, centers_batch), dim=1)
            features_shift = torch.cat((h, centers_batch), dim=1)
            p_shift = self.fc_shift(features_shift)

            # dists = torch.cdist(features_class, self.centers, p=2)
            # sim = 1 / (1 + dists)
            # p_shift = self.fc_shift(sim)
            p_shift = torch.clamp(p_shift, min=self.config.min_shift, max=self.config.max_shift)  # 输出限制

            # 预测
            # pred = self.fusion(h)
            return pred, pred_center_score, p_class, p_shift

        if mode == "tsne":
            return features_class



    def forward(self, video, acoustic, lengths, label_area, mode):
        batch_size = lengths.size(0)
        if mode == "train":
            pred, pred_center_score, p_class, p_shifting, uncertainty, order_center_loss = self.alignment(video, acoustic, lengths, label_area, mode)
            return pred, pred_center_score, p_class, p_shifting, uncertainty, order_center_loss

        if mode == "test":
            pred, pred_center_score, p_class, p_shifting = self.alignment(video, acoustic, lengths, label_area, mode)
            return pred, pred_center_score, p_class, p_shifting

        if mode == "tsne":
            features_class = self.alignment(video, acoustic, lengths, label_area, mode)
            return features_class