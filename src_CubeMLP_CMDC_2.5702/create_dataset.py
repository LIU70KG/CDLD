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

def pad_to_length(arr, target_len=12):
    curr_len, feature_dim = arr.shape
    if curr_len < target_len:
        pad_len = target_len - curr_len
        pad = np.zeros((pad_len, feature_dim), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    elif curr_len > target_len:
        arr = arr[:target_len]  # 可选：截断
    return arr


class CMDC:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            data = load_pickle(DATA_PATH + '/' + config.cross_validation + '.pkl')
            describe = 'describe: (words, visual, acoustic, wordtxt), label_PHQ-9, describe)'
            self.train = []
            for (ex_index, example) in enumerate(data["train"]):
                (visual, acoustic, words), (label_id_class, label_id), wordtxt = example
                visual = pad_to_length(visual, target_len=12)
                acoustic = pad_to_length(acoustic, target_len=12)
                words = pad_to_length(words, target_len=12)
                data_tuple = [((words, visual, acoustic, wordtxt), np.array([[float(label_id)]], dtype=np.float32), describe)]
                self.train.extend(data_tuple)
            self.dev = data["valid"]
            self.test = []
            for (ex_index, example) in enumerate(data["test"]):
                (visual, acoustic, words), (label_id_class, label_id), wordtxt = example
                visual = pad_to_length(visual, target_len=12)
                acoustic = pad_to_length(acoustic, target_len=12)
                words = pad_to_length(words, target_len=12)
                data_tuple = [((words, visual, acoustic, wordtxt), np.array([[float(label_id)]], dtype=np.float32), describe)]
                self.test.extend(data_tuple)

            self.word2id = None
            self.pretrained_emb = None

            # 继续写，调整数据顺序，和mosi一样的

        except:
            print("N0 CMDC file")

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
            labels = [sample[1][0] for sample in self.train]

        elif mode == "dev":
            labels = [sample[1][0] for sample in self.train]

        elif mode == "test":
            labels = [sample[1][0] for sample in self.test]

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