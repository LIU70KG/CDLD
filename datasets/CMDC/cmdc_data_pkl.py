# 作者：刘成广
# 时间：2024/3/8 下午8:39
import pickle
import os
import glob
import pandas as pd
import numpy as np
import re

# 数据路径
dataset_paths = '../../dataset/'  # 申请的数据集路径
# 读取 Excel 文件
df = pd.read_excel('../../SubjectInfo.xlsx')  # 申请的数据集里附带数据信息
date_information = pd.concat([df['ID'], df['MDD'], df['VideoAvailabe'], df['PHQtotal']], axis=1)
# # ----------------------45个3模态数据读取-------------------------
# 找出 VideoAvailabe 可用的数据
data_video_available = date_information[date_information['VideoAvailabe'] == 1]
data_VAT_modal = []
for index, row in data_video_available.iterrows():
    date_ = row.values.tolist()  # df['ID'], df['MDD'], df['VideoAvailabe'], df['PHQtotal']
    print(f"正在处理数据： {date_[0]}.")
    # 找到Q*.npy的视觉深度表示（通过在Kinetics 600上预训练的时间成型器提取），按照问题排序后读取特征
    deepCV_fea_file = glob.glob(os.path.join(dataset_paths, date_[0], '*.npy'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in deepCV_fea_file]
    deepCV_fea_file_sort = [file for _, file in sorted(zip(numbers, deepCV_fea_file))]
    data_list = [np.load(file) for file in deepCV_fea_file_sort]
    deepCV_fea = np.vstack(data_list)

    # 找到Q*.pkl:(VGGish提取的音频特征)，按照问题排序后读取特征
    deepAudio_fea_file = glob.glob(os.path.join(dataset_paths, date_[0], '*.pkl'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in deepAudio_fea_file]
    deepAudio_fea_file_sort = [file for _, file in sorted(zip(numbers, deepAudio_fea_file))]
    data_list = [pickle.load(open(file, 'rb')) for file in deepAudio_fea_file_sort]
    data_list_no_nan = []
    for arr in data_list:
        if np.isnan(arr).any():
            arr = np.zeros((arr.shape[0], arr.shape[1]))
            print(f"音频特征存在nan, 补0该问题Qx特征")
        data_list_no_nan.extend([arr])
    deepAudio_fea = np.vstack(data_list_no_nan)

    # 找到文本Q*的音频流转录，并排序
    txt_file = glob.glob(os.path.join(dataset_paths, date_[0], '*.txt'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in txt_file]
    txt_file_sort = [file for _, file in sorted(zip(numbers, txt_file))]
    data_list = [open(file, 'r').read() for file in txt_file_sort]
    Audio_txt = np.squeeze(np.vstack(data_list))

    # 找到文本的bert嵌入，按照问题排序后读取特征
    deeptxt_fea_file = glob.glob(os.path.join(dataset_paths, date_[0], 'vector_Q*.csv'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in deeptxt_fea_file]
    deeptxt_fea_file_sort = [file for _, file in sorted(zip(numbers, deeptxt_fea_file))]
    # data_list = [np.array(pd.read_csv(file).columns.tolist(), dtype=np.float32) for file in deeptxt_fea_file_sort]
    data_list = []
    for file in deeptxt_fea_file_sort:
        deeptxt_fea_str = pd.read_csv(file).columns.tolist()
        # deeptxt_fea_str 是list，里面每一个元素都是str，在这里添加代码，使得忽略非第一个的小数点，并deeptxt_fea_str继续转化为np.float32
        ttt_list = []
        for column_name in deeptxt_fea_str:
            # 使用正则表达式匹配浮点数格式的字符串，提取第一个匹配项
            match = re.search(r'[-+]?\d*\.\d+', column_name)
            # 如果找到了匹配项，则提取第一个匹配项并转换为 np.float32 类型
            if match:
                float_value = np.float32(match.group())
                ttt_list.append(float_value)
            else:
                print("No floating point number found in the column name.")
        data_list.extend([ttt_list])
    deeptxt_fea = np.squeeze(np.vstack(data_list))
    data_tuple = [((deepCV_fea, deepAudio_fea, deeptxt_fea), (date_[1], date_[3]), Audio_txt)]
    data_VAT_modal.extend(data_tuple)

train_data1 = data_VAT_modal[0:19+1] + data_VAT_modal[26:41]
test_data1 = data_VAT_modal[20:25+1] + data_VAT_modal[41:44]

train_data2 = data_VAT_modal[6:25+1] + data_VAT_modal[29:44]
test_data2 = data_VAT_modal[0:5+1] + data_VAT_modal[26:28+1]

train_data3 = data_VAT_modal[0:5+1] + data_VAT_modal[12:25+1] + data_VAT_modal[26:28+1] + data_VAT_modal[32:44]
test_data3 = data_VAT_modal[6:11+1] + data_VAT_modal[29:31+1]

train_data4 = data_VAT_modal[0:11+1] + data_VAT_modal[18:25+1] + data_VAT_modal[26:31+1] + data_VAT_modal[35:44]
test_data4 = data_VAT_modal[12:17+1] + data_VAT_modal[32:34+1]

train_data5 = data_VAT_modal[0:17+1] + data_VAT_modal[24:25+1] + data_VAT_modal[26:34+1] + data_VAT_modal[37:44]
test_data5 = data_VAT_modal[18:23+1] + data_VAT_modal[35:37]

# 构建数据结构
data1 = {"train": train_data1, "valid": "数据太少，不提供", "test": test_data1, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data2 = {"train": train_data2, "valid": "数据太少，不提供", "test": test_data2, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data3 = {"train": train_data3, "valid": "数据太少，不提供", "test": test_data3, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data4 = {"train": train_data4, "valid": "数据太少，不提供", "test": test_data4, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data5 = {"train": train_data5, "valid": "数据太少，不提供", "test": test_data5, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}


# 保存为.pkl文件
output_file = 'cmdc_data_all_modal_1.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data1, file)
print(f"data_VAT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_all_modal_2.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data2, file)
print(f"data_VAT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_all_modal_3.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data3, file)
print(f"data_VAT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_all_modal_4.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data4, file)
print(f"data_VAT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_all_modal_5.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data5, file)
print(f"data_VAT_modal has been saved as {output_file}.")
# ----------------------78个2模态数据读取-------------------------
data_AT_modal = []
for index, row in date_information.iterrows():
    date_ = row.values.tolist()
    print(f"正在处理数据： {date_[0]}.")
    # 找到Q*.pkl:(VGGish提取的音频特征)，按照问题排序后读取特征
    deepAudio_fea_file = glob.glob(os.path.join(dataset_paths, date_[0], '*.pkl'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in deepAudio_fea_file]
    deepAudio_fea_file_sort = [file for _, file in sorted(zip(numbers, deepAudio_fea_file))]
    data_list = [pickle.load(open(file, 'rb')) for file in deepAudio_fea_file_sort]
    data_list_no_nan = []
    for arr in data_list:
        if np.isnan(arr).any():
            arr = np.zeros((arr.shape[0], arr.shape[1]))
            print(f"音频特征存在nan, 补0该问题Qx特征")
        data_list_no_nan.extend([arr])
    deepAudio_fea = np.vstack(data_list_no_nan)

    # 找到文本Q*的音频流转录，并排序
    txt_file = glob.glob(os.path.join(dataset_paths, date_[0], '*.txt'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in txt_file]
    txt_file_sort = [file for _, file in sorted(zip(numbers, txt_file))]
    data_list = [open(file, 'r').read() for file in txt_file_sort]
    Audio_txt = np.squeeze(np.vstack(data_list))

    # 找到文本的bert嵌入，按照问题排序后读取特征
    deeptxt_fea_file = glob.glob(os.path.join(dataset_paths, date_[0], 'vector_Q*.csv'))
    numbers = [int(filename.split('Q')[-1].split('.')[0]) for filename in deeptxt_fea_file]
    deeptxt_fea_file_sort = [file for _, file in sorted(zip(numbers, deeptxt_fea_file))]
    data_list = []
    for file in deeptxt_fea_file_sort:
        deeptxt_fea_str = pd.read_csv(file).columns.tolist()
        # deeptxt_fea_str 是list，里面每一个元素都是str，在这里添加代码，使得忽略非第一个的小数点，并deeptxt_fea_str继续转化为np.float32
        ttt_list = []
        for column_name in deeptxt_fea_str:
            # 使用正则表达式匹配浮点数格式的字符串，提取第一个匹配项
            match = re.search(r'[-+]?\d*\.\d+', column_name)
            # 如果找到了匹配项，则提取第一个匹配项并转换为 np.float32 类型
            if match:
                float_value = np.float32(match.group())
                ttt_list.append(float_value)
            else:
                print("No floating point number found in the column name.")
        data_list.extend([ttt_list])
    deeptxt_fea = np.squeeze(np.vstack(data_list))
    data_tuple = [((deepAudio_fea, deeptxt_fea), (date_[1], date_[3]), Audio_txt)]
    data_AT_modal.extend(data_tuple)

train_data1 = data_AT_modal[0:39+1] + data_AT_modal[52:71+1]
test_data1 = data_AT_modal[40:51+1] + data_AT_modal[72:77+1]

train_data2 = data_AT_modal[12:51+1] + data_AT_modal[58:77+1]
test_data2 = data_AT_modal[0:11+1] + data_AT_modal[52:57+1]

train_data3 = data_AT_modal[0:11+1] + data_AT_modal[24:51+1] + data_AT_modal[52:57+1] + data_AT_modal[64:77+1]
test_data3 = data_AT_modal[12:23+1] + data_AT_modal[58:63+1]

train_data4 = data_AT_modal[0:23+1] + data_AT_modal[36:51+1] + data_AT_modal[52:63+1] + data_AT_modal[70:77+1]
test_data4 = data_AT_modal[24:35+1] + data_AT_modal[64:69+1]

train_data5 = data_AT_modal[0:35+1] + data_AT_modal[48:51+1] + data_AT_modal[52:69+1] + data_AT_modal[76:77+1]
test_data5 = data_AT_modal[36:47+1] + data_AT_modal[70:75+1]

# 构建数据结构
data1 = {"train": train_data1, "valid": "数据太少，不提供", "test": test_data1, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data2 = {"train": train_data2, "valid": "数据太少，不提供", "test": test_data2, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data3 = {"train": train_data3, "valid": "数据太少，不提供", "test": test_data3, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data4 = {"train": train_data4, "valid": "数据太少，不提供", "test": test_data4, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}
data5 = {"train": train_data5, "valid": "数据太少，不提供", "test": test_data5, "description": "((deepCV_fea, deepAudio_fea, deeptxt_fea), date_[3]:标签, Audio_txt)"}

# 保存为.pkl文件
output_file = 'cmdc_data_AT_modal_1.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data1, file)
print(f"data_AT_modal has been saved as {output_file}.")

# 保存为.pkl文件
output_file = 'cmdc_data_AT_modal_2.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data2, file)
print(f"data_AT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_AT_modal_3.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data3, file)
print(f"data_AT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_AT_modal_4.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data4, file)
print(f"data_AT_modal has been saved as {output_file}.")

output_file = 'cmdc_data_AT_modal_5.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(data5, file)
print(f"data_AT_modal has been saved as {output_file}.")