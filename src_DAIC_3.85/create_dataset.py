import sys
import mmsdk
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError
from collections import Counter
import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)




class DAIC_WOZ:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:

            self.dev = load_pickle(DATA_PATH + '/valid_data_paragraph_concat.pkl')['valid']  # 35
            self.train = load_pickle(DATA_PATH + '/train_valid_data_paragraph_concat.pkl')['train_valid']  # 142
            self.test = load_pickle(DATA_PATH + '/test_data_paragraph_concat.pkl')['test']  # 47
            self.word2id = None
            self.pretrained_emb = None

        except:
            print("N0 DAIC_WOZ file")

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


    def get_shample_number(self, mode):

        if mode == "train":
            labels = [sample[1][1] for sample in self.train]

        elif mode == "dev":
            labels = [sample[1][1] for sample in self.dev]

        elif mode == "test":
            labels = [sample[1][1] for sample in self.test]

        else:
            print("Mode is not set properly (train/dev/test/train_dev)")
            exit()

        labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        label_area = torch.floor_divide(labels, 5)
        label_area[labels >= 25] = 4  # 处理标签在25及以上的情况
        label_area = label_area.squeeze().int().tolist()
        counter = Counter(label_area)
        shample_number = [v for k, v in sorted(counter.items())]
        return shample_number