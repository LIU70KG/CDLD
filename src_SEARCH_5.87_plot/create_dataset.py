import pickle
import numpy as np
from collections import Counter
import torch


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def calculate_labels(scores):
    labels = []
    for score in scores:
        if score <= 9:
            labels.append(0)  # 正常
        elif 10 <= score <= 13:
            labels.append(1)  # 轻度
        elif 14 <= score <= 20:
            labels.append(2)  # 中度
        elif 21 <= score <= 27:
            labels.append(3)  # 重度
        else:
            labels.append(4)  # 非常严重
    return labels


class SEARCH:
    def __init__(self, config):
        self.config = config
        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + 'train.pkl')
            self.test = load_pickle(DATA_PATH + 'test.pkl')
        except:
            print("N0 SEARCH file")


    def calculate_Average_interval(self, scores):
        labels = []
        if self.config.interval_Scale == 'average' and self.config.interval_num == 6:
            for score in scores:
                if score <= 6:
                    labels.append(0)  # 正常1
                elif 7 <= score <= 13:
                    labels.append(1)  # 正常2
                elif 14 <= score <= 21:
                    labels.append(2)  # 轻度
                elif 22 <= score <= 29:
                    labels.append(3)  # 中度
                elif 30 <= score <= 37:
                    labels.append(4)  # 严重1
                else:
                    labels.append(5)  # 严重2
        elif self.config.interval_Scale == 'standards' and self.config.interval_num == 6:
            for score in scores:
                if score <= 9:
                    labels.append(0)  # 正常
                elif 10 <= score <= 13:
                    labels.append(1)  # 轻度
                elif 14 <= score <= 20:
                    labels.append(2)  # 中度
                elif 21 <= score <= 27:
                    labels.append(3)  # 中重
                elif 28 <= score <= 36:
                    labels.append(4)  # 严重1
                else:
                    labels.append(5)  # 严重2
        elif self.config.interval_Scale == 'average' and self.config.interval_num == 5:
            for score in scores:
                if score <= 9:
                    labels.append(0)  # 正常
                elif 10 <= score <= 18:
                    labels.append(1)  # 轻度
                elif 19 <= score <= 27:
                    labels.append(2)  # 中度
                elif 28 <= score <= 36:
                    labels.append(3)  # 中重
                else:
                    labels.append(4)  # 严重
        elif self.config.interval_Scale == 'standards' and self.config.interval_num == 5:
            for score in scores:
                if score <= 9:
                    labels.append(0)  # 正常
                elif 10 <= score <= 13:
                    labels.append(1)  # 轻度
                elif 14 <= score <= 20:
                    labels.append(2)  # 中度
                elif 21 <= score <= 27:
                    labels.append(3)  # 中重
                else:
                    labels.append(4)  # 严重
        return labels

    def get_shample_number(self, mode):

        if mode == "train":
            score = [sample[3] for sample in self.train]
            mut_class_labels = [sample[4] for sample in self.train]
        elif mode == "test":
            score = [sample[3] for sample in self.test]
            mut_class_labels = [sample[4] for sample in self.train]
        else:
            print("Mode is not set properly (train/test)")
            exit()

        # mut_class_labels = calculate_labels(score)
        counter = Counter(mut_class_labels)  # Counter({0: 1972, 2: 792, 1: 689, 3: 222, 4: 206})3881

        # labels = torch.tensor(score, dtype=torch.float32).view(-1, 1)
        # label_area = torch.floor_divide(labels, 5)
        # label_area[labels >= 25] = 4  # 处理标签在25及以上的情况
        # label_area = label_area.squeeze().int().tolist()
        # counter = Counter(label_area)  # Counter({0: 1972, 2: 1023, 4: 528, 3: 358})

        shample_number = [v for k, v in sorted(counter.items())]
        return shample_number


    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()
