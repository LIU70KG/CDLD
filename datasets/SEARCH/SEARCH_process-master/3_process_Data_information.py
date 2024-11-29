# 按照医院提供的平衡性别年龄的数据文件，将3个地区的数据，分主题保存
# 保存形式：id、cv_fileName、cv_question、audio_fileName、audio_question、score、level
# 例如：6954 /home/liu70kg/D512G/SEARCH/vision/sheyang_features/read_video_1301612010070320000_6954_52_1_40_1667180302117.csv 0
# /home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_feature_read/sheyang/read_video_1301612010070320000_6954_52_1_40_1667180302117.npy 0 0 0


import os
import pandas as pd
import random
import glob
import re
import numpy as np
import torch
import pickle
from tqdm import tqdm

subject = ['depressed', 'anxiety', 'pressure', 'sleepless', 'selfhurt']
xlsxPath = '/home/liu70kg/PycharmProjects/Depression/SEARCH/balance_data_Selection.xlsx'  # 医院提供的平衡性别年龄的数据文件，标注疾病等级(轻中重)标签的患者组和健康组
audio_datapath = ['/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_feature_read/sheyang',
            '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_feature_read/taizhou',
            '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_feature_read/yixing']

cv_datapath = ['/home/liu70kg/D512G/SEARCH/vision/sheyang_features',
            '/home/liu70kg/D512G/SEARCH/vision/taizhou_features',
            '/home/liu70kg/D512G/SEARCH/vision/yixing_features']

feature_choice = 'AU+Gaze+Rigid+Pose'

# 创建字典，保存 样本序号id 及 文件路径fileName 及question(0、1、2)
cv_dict = {}
audio_dict = {}


def add_cv_dict(id, fileName, question):
    if id not in cv_dict:
        cv_dict[id] = []  # 如果邮编不在字典中，创建一个新的列表
    cv_dict[id].append({'id': id, 'cv_fileName': fileName, 'cv_question': question})


def add_audio_dict(id, fileName, question):
    if id not in audio_dict:
        audio_dict[id] = []  # 如果邮编不在字典中，创建一个新的列表
    audio_dict[id].append({'id': id, 'audio_fileName': fileName, 'audio_question': question})


class0 = 300  # 每个地区的健康样本只取300个
for area_index, cv_dataPath_part in enumerate(cv_datapath):
    # ------------------视频文件-------------------
    cv_dataList = glob.glob(os.path.join(cv_dataPath_part, '*.csv'))
    for fileName in cv_dataList:
        elements = fileName.split('/')[-1].split('_')
        # 找到由4到5位数数字组成的元素及其序号
        result = [(index, elem) for index, elem in enumerate(elements) if re.fullmatch(r'\d{4,5}', elem)]
        if not result:
            continue
        id = int(result[0][1])
        question = elements[int(result[0][0]) + 2]
        if elements[0] == 'read':
            question = '0'

        add_cv_dict(id, fileName, question)

    # ------------------音频文件-------------------
    audio_dataPath_part = audio_datapath[area_index]
    audio_dataList = glob.glob(os.path.join(audio_dataPath_part, '*.npy'))
    for fileName in audio_dataList:
        elements = fileName.split('/')[-1].split('_')
        # 找到由4到5位数数字组成的元素及其序号
        result = [(index, elem) for index, elem in enumerate(elements) if re.fullmatch(r'\d{4,5}', elem)]
        if not result:
            continue
        id = int(result[0][1])
        question = elements[int(result[0][0]) + 2]
        if elements[0] == 'read':
            question = '0'

        add_audio_dict(id, fileName, question)

    def get(path, frame, feature_choice):
        # 选择特征
        # 创建一个空字典, 保存要提取的特征
        features = {}
        csvFile = pd.read_csv(path)

        if 'AU' in feature_choice.split('+'):
            # AU = csvFile.iloc[indices, 679: 714]  # 35
            AU = csvFile.iloc[frame, 679: 696]
            features["AU"] = torch.tensor(AU.values, dtype=torch.float32)
        if 'Gaze' in feature_choice.split('+'):
            Gaze = csvFile.iloc[frame, 5:13]
            # Gaze = csvFile.iloc[frame, 5:5 + 3 + 3 + 2 + 112] # 5: 13  8
            # Gaze = csvFile.iloc[frame, 5: 293]
            features["Gaze"] = torch.tensor(Gaze.values, dtype=torch.float32)
        if 'Pose' in feature_choice.split('+'):
            Pose = csvFile.iloc[frame, 293:293 + 3 + 3]  # 6
            # Pose = csvFile.iloc[frame, 293:293 + 3 + 3 + 136]  # 6
            # Pose = csvFile.iloc[frame, 293: 639]
            features["Pose"] = torch.tensor(Pose.values, dtype=torch.float32)
        if 'Rigid' in feature_choice.split('+'):
            #  feature = csvFile.iloc[indices, 639: 643]  # 4
            Rigid = csvFile.iloc[frame, 639: 679]  # 639: 642
            features["Rigid"] = torch.tensor(Rigid.values, dtype=torch.float32)
        if 'manual' in feature_choice.split('+'):
            manual = csvFile.iloc[frame, 293 + 6: 68 * 2 + 293 + 6], csvFile.iloc[frame, 679: 696]
            features["manual"] = torch.tensor(np.concatenate((manual), axis=1), dtype=torch.float32)
        if 'features_all' in feature_choice.split('+'):
            features_all = csvFile.iloc[frame, 5: 714]
            features["features_all"] = torch.tensor(features_all.values, dtype=torch.float32)
            # features["features_all"] = self.pca.fit_transform(features["features_all"])


        feature=[]
        for key in feature_choice.split('+'):
            feature.append(features[key])
        features_choose = torch.tensor(np.concatenate(feature, axis=1), dtype=torch.float32)
        cv_features = features_choose.reshape(int(features_choose.shape[0]/3), int(features_choose.shape[1]*3))
        return cv_features.numpy()


for i in range(0, 1):  # 只取特定表格,这里只取抑郁样本信息
    sheetName = i  # 0-4分别对应xlsx里的5张表
    if (sheetName <= 3):
        xlsx1File = pd.read_excel(xlsxPath, sheet_name=sheetName, usecols=[0, 1, 2], names=['ID', '分数', '等级'])
        xlsx2File = pd.read_excel(xlsxPath, sheet_name=sheetName, usecols=[3, 4, 5], names=['ID', '分数', '等级'])
    elif (sheetName == 4):  # 自伤没有分数，所以复制等级为分数（即对分数列进行重复）
        xlsx1File = pd.read_excel(xlsxPath, sheet_name=sheetName, usecols=[0, 1], names=['ID', '分数'])
        xlsx1File.insert(2, '等级', xlsx1File['分数'])
        xlsx2File = pd.read_excel(xlsxPath, sheet_name=sheetName, usecols=[2, 3], names=['ID', '分数'])
        xlsx2File.insert(2, '等级', xlsx2File['分数'])

    merge_xlsx = pd.concat([xlsx1File, xlsx2File], ignore_index=True)  # 原xlsx文件分成了两列进行对照，这里需要竖着拼起来方便根据文件名找id
    merge_xlsx.sort_values(by=['ID'], inplace=True)
    merge_xlsx = merge_xlsx.reset_index(drop=True)  # 排序
    ids = merge_xlsx.iloc[:, 0].values  # 数据中的第0列，也就是所有的id

    txtPath = '/home/liu70kg/PycharmProjects/Depression/SEARCH/{subject}_BalancedDataList.txt'.format(
        subject=subject[sheetName])  # ------0- 4---------------------
    SEARCH_fea_all = []
    with open(txtPath, 'w', encoding="utf-8") as F:
        lineS = 'id、cv_fileName、cv_question、audio_fileName、audio_question、score、level\n'
        F.write(lineS)
        # for index, balance_data_Selection_id in enumerate(ids):
        # ids = ids.tolist()
        # 使用 tqdm 包裹 enumerate，并显示进度条
        for index, balance_data_Selection_id in tqdm(enumerate(ids), total=len(ids), desc="Processing IDs"):
            if balance_data_Selection_id in cv_dict and balance_data_Selection_id in audio_dict:
                score = merge_xlsx.iloc[index, 1]
                level = merge_xlsx.iloc[index, 2]
                cv_file = cv_dict[balance_data_Selection_id]
                audio_file = audio_dict[balance_data_Selection_id]
                for pp, cv_file_x in enumerate(cv_file):
                   id = cv_file_x['id']
                   cv_fileName = cv_file_x['cv_fileName']
                   cv_question = cv_file_x['cv_question']
                   elements1 = cv_fileName.split('/')[-1].split('_')
                   # ----------防止一个id对应多个样本数据----------
                   try:
                       index = elements1.index(str(id))  # 查找目标数字的位置
                       mark_cv =  elements1[index + 1:index + 4]  # 获取后面的三个元素
                       flag = 0
                       for audio_file1 in audio_file:
                           audio_fileName = audio_file1['audio_fileName']
                           audio_question = audio_file1['audio_question']
                           elements1 = audio_fileName.split('/')[-1].split('_')
                           index = elements1.index(str(id))
                           mark_audio =  elements1[index + 1:index + 4]

                           if mark_cv != mark_audio:
                               continue
                           else:
                               flag = 1
                               break

                   except (ValueError, IndexError):
                       continue

                   if flag == 0:
                       continue
                   #  整理需要用的数据
                   audio_feature = np.load(audio_fileName) # 读取 .npy 文件
                   seq_length = audio_feature.shape[0]
                   # -------------------------阅读时间少于10秒，都是出问题的。北风与太阳至少需要20S
                   if seq_length < 10:
                       continue

                   frame = [int(p * 9.6) + 1 for p in range(seq_length * 3)]
                   # if cv_fileName == "/home/liu70kg/D512G/SEARCH/vision/sheyang_features/read_video_1301612008090120800_8554_52_1_37_1667289991518.csv":
                   #     print(frame)
                   #     print(cv_fileName)
                   #     print(audio_fileName)
                   try:
                       cv_feature = get(cv_fileName, frame, feature_choice)  # id=37951有问题，视频将22秒，语音45秒
                   except (ValueError, IndexError):
                       print("视频与语音不对应的样本： \n", id)
                       print(audio_fileName)
                   shample_inforamtion = [(cv_feature, audio_feature, id, score, level)]
                   SEARCH_fea_all.extend(shample_inforamtion)

                   lineS = '{id} {cv_fileName} {cv_question} {audio_fileName} {audio_question} {score} {level}\n'.format(id=id,
                                                                                                cv_fileName=cv_fileName,
                                                                                                cv_question=cv_question,
                                                                                                audio_fileName=audio_fileName,
                                                                                                audio_question=audio_question,
                                                                                                score=score,
                                                                                                level=level)
                   F.write(lineS)


    description = "解释：( 视觉特征(/0.96s), 语音特征(/0.96s), 样本编号，分数，类别 ）"
    SEARCH_fea_all_file = '/home/liu70kg/PycharmProjects/Depression/SEARCH/SEARCH_fea_all_concat.pkl'
    SEARCH_fea = {"all": SEARCH_fea_all, "description": description}
    with open(SEARCH_fea_all_file, 'wb') as file:
        pickle.dump(SEARCH_fea, file)
    print(f"数据集SEARCH全部部分 has been saved as {SEARCH_fea_all_file}.")