# 将数据划分为训练集和测试集
from glob import glob
import os
import torch
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from collections import Counter


SEARCH_fea_all_file = '/home/liu70kg/PycharmProjects/Depression/SEARCH/SEARCH_fea_all_concat.pkl'
SEARCH_fea_train_test = '/home/liu70kg/PycharmProjects/Depression/SEARCH/SEARCH'

with open(SEARCH_fea_all_file, 'rb') as f:
    data_feature_all = pickle.load(f)['all']  # 从打开的文件中加载数据
mut_class_labels = [sample[4] for sample in data_feature_all]
counter = Counter(mut_class_labels)
print(f"每个类别样本的个数: {counter}")
# # 分割为 8:2 的训练集和测试集
# train_data, test_data = train_test_split(data_feature_all, test_size=0.2, random_state=42)

# 初始化 KFold，设置 n_splits=5 表示 5 折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 将 data_feature_all 转换为 numpy 数组以便分割
data_feature_all = np.array(data_feature_all, dtype=object)
mut_class_labels = np.array(mut_class_labels)
# 循环获取每折的训练集和测试集索引，并保存到文件
for fold, (train_index, test_index) in enumerate(skf.split(data_feature_all, mut_class_labels), 1):  # 从1开始计数
    print(f"Fold {fold}:")
    train_data, test_data = data_feature_all[train_index], data_feature_all[test_index]
    train_labels, test_labels = mut_class_labels[train_index], mut_class_labels[test_index]
    train_data = train_data.tolist()
    test_data = test_data.tolist()

    # 统计训练集和测试集的类别分布
    train_counter = Counter(train_labels)
    test_counter = Counter(test_labels)
    # 输出训练集和测试集的类别分布
    print(f"训练集的类别分布: {train_counter}")
    print(f"测试集的类别分布: {test_counter}")

    # 定义文件名
    train_filename = f'SEARCH_data_VA_fold_{fold}_train.pkl'
    test_filename = f'SEARCH_data_VA_fold_{fold}_test.pkl'

    # 完整的文件路径
    train_filepath = os.path.join(SEARCH_fea_train_test, train_filename)
    test_filepath = os.path.join(SEARCH_fea_train_test, test_filename)

    # 保存 train_data 到文件
    with open(train_filepath, 'wb') as f_train:
        pickle.dump(train_data, f_train)

    # 保存 test_data 到文件
    with open(test_filepath, 'wb') as f_test:
        pickle.dump(test_data, f_test)

    print(f"训练集已保存到: {train_filepath}")
    print(f"测试集已保存到: {test_filepath}")
    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")


