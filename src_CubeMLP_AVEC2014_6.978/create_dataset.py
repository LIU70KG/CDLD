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
        if score <= 13:
            labels.append(0)  # 正常
        elif 14 <= score <= 19:
            labels.append(1)  # 轻度
        elif 20 <= score <= 28:
            labels.append(2)  # 中度
        else:
            labels.append(3)  # 非常严重
    return labels



class AVEC2014:
    def __init__(self, config):
        self.config = config
        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            # self.train = load_pickle(DATA_PATH + 'train.pkl')
            # self.dev = load_pickle(DATA_PATH + 'dev.pkl')
            self.test = load_pickle(DATA_PATH + 'test.pkl')
            self.train = load_pickle(DATA_PATH + 'train_dev.pkl')
            self.all = load_pickle(DATA_PATH + 'all.pkl')
        except:
            print("N0 AVEC2014 file")

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
                if score <= 6:
                    labels.append(0)  # 正常1
                elif 7 <= score <= 13:
                    labels.append(1)  # 正常2
                elif 14 <= score <= 19:
                    labels.append(2)  # 轻度
                elif 20 <= score <= 28:
                    labels.append(3)  # 中度
                elif 29 <= score <= 37:
                    labels.append(4)  # 严重1
                else:
                    labels.append(5)  # 严重2
        elif self.config.interval_Scale == 'average' and self.config.interval_num == 5:
            for score in scores:
                if score <= 8:
                    labels.append(0)  # 正常
                elif 9 <= score <= 17:
                    labels.append(1)  # 轻度
                elif 18 <= score <= 26:
                    labels.append(2)  # 中度
                elif 27 <= score <= 35:
                    labels.append(3)  # 严重1
                else:
                    labels.append(4)  # 严重2
        elif self.config.interval_Scale == 'standards' and self.config.interval_num == 5:
            for score in scores:
                if score <= 13:
                    labels.append(0)  # 正常
                elif 14 <= score <= 19:
                    labels.append(1)  # 轻度
                elif 20 <= score <= 28:
                    labels.append(2)  # 中度
                elif 29 <= score <= 37:
                    labels.append(3)  # 严重1
                else:
                    labels.append(4)  # 严重2
        return labels


    def get_shample_number(self, mode):

        if mode == "train":
            score = [sample[3] for sample in self.train]
            # mut_class_labels = [sample[4] for sample in self.train]
            mut_class_labels = self.calculate_Average_interval(score)
        elif mode == "test":
            score = [sample[3] for sample in self.test]
            # mut_class_labels = [sample[4] for sample in self.train]
            mut_class_labels = self.calculate_Average_interval(score)
        else:
            print("Mode is not set properly (train/test)")
            exit()

        # mut_class_labels = calculate_labels(score)
        counter = Counter(mut_class_labels)  # Counter({0: 33, 1: 19, 3: 18, 4: 13, 2: 12, 5: 5})
        score_all = [sample[3] for sample in self.all]
        mut_class_all = [sample[4] for sample in self.all]
        a = {sample[2]: sample[3] for sample in self.all}
        sorted_a_dict = dict(sorted(a.items(), key=lambda item: item[1]))
        # 训练 + 验证：Counter({0: 52, 3: 18, 2: 18, 1: 12})
        # 测试：Counter({0: 77, 2: 26, 3: 25, 1: 22})
        # score_all其实最小0,最大45

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
