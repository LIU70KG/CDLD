import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
# from transformers import *
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup, BertModel, BertConfig
from create_dataset import SEARCH
from numpy.random import randint
bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased/')  # 注意此处为本地文件夹, 下载bert-base-uncased到ILOC下


class MSADataset(Dataset):
    def __init__(self, config):
        dataset = SEARCH(config)
        self.mode = config.mode
        self.num_segments = 10
        self.duration = 2
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)
        shample_number = dataset.get_shample_number(config.mode)
        # 计算权重（样本数的倒数）
        shample_number = torch.tensor(shample_number, dtype=torch.float32)
        weights = 1.0 / shample_number
        config.weights = weights / weights.sum()  # 正则化权重，使其和为 1
        config.visual_size = self.data[0][0].shape[1]
        config.acoustic_size = self.data[0][1].shape[1]


    def _get_fragment(self, record):
        # # split all frames into seg parts, then select frame in each part
        # return record
        num_frames = record[0].shape[0]
        if num_frames >= self.num_segments * self.duration * 2:
            step = num_frames // (self.num_segments * self.duration)  # 步长
            offsets = np.arange(0, (self.num_segments * self.duration) * step, step)[:20]
        elif self.num_segments * self.duration * 2 > num_frames > self.num_segments * self.duration:
            # 如果 40 > x > 20，随机选择 20 个不重复的整数
            offsets = np.sort(np.random.choice(range(num_frames), size=self.num_segments * self.duration, replace=False))
        else:
            # 如果 x <= 20
            # selection = np.arange(0, num_frames)
            # extra_needed = self.num_segments * self.duration - len(selection)
            # extra = np.full(extra_needed, num_frames - 1, dtype=int)
            # offsets = np.sort(np.concatenate([selection, extra]))

            offsets = np.arange(0, num_frames)


        # 挑选帧
        visual_fea, audio_fea, number, score, label = record
        visual = visual_fea[offsets, :]
        audio = audio_fea[offsets, :]
        paragraph_inforamtion = (visual, audio, number, score, label)
        return paragraph_inforamtion


    def __getitem__(self, index):
        record = self.data[index]

        if self.mode == 'train':
            segment = self._get_fragment(record)
            return segment
        elif self.mode == 'test':
            segment = self._get_fragment(record)
            return segment

        # return self.data[index]


    def __len__(self):
        return self.len


def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    calculate_Average_interval = SEARCH(config).calculate_Average_interval

    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        score = [sample[3] for sample in batch]
        labels = torch.tensor(score, dtype=torch.float32).view(-1, 1)
        # label_area = calculate_labels(score)  # mut_class_labels
        # label_area = [sample[4] for sample in batch]
        label_area = calculate_Average_interval(score)
        center_score = config.center_score
        center_score_values = torch.tensor([center_score[i] for i in label_area], dtype=torch.float32).view(-1, 1)

        score = torch.tensor(score, dtype=torch.float32).view(-1, 1)
        label_area = torch.tensor(label_area, dtype=torch.float32).view(-1, 1)
        label_shifting = score - center_score_values  # max:10, min =-4

        # labels = torch.tensor(score, dtype=torch.float32).view(-1, 1)
        # label_area = torch.floor_divide(labels, 5)
        # label_area[labels >= 25] = 4  # 处理标签在25及以上的情况
        # label_shifting = labels - (label_area * 5 + 2)

        visual = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])
        return visual, acoustic, labels, label_area, label_shifting, lengths


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
