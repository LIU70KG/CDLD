# 作者：刘成广
# 时间：2024/9/29 下午7:57
# 使用frcrn，给音频文件去除背景噪声
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from tqdm import tqdm

dataSource = '/home/liu70kg/D512G/SEARCH/audio'
dataOutDir = '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio'
folderName = ['sheyang', 'taizhou', 'yixing']
# folderName = ['taizhou', 'yixing']
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
for place in folderName:
    dataSource1 = os.path.join(dataSource, place)
    dataOutDir1 = os.path.join(dataOutDir, place)

    # 音频在某文件夹内
    wav_list = os.listdir(dataSource1)
    for filename in tqdm(wav_list, desc='去噪'):
        wav_path = os.path.join(dataSource1, filename)
        outputPath = os.path.join(dataOutDir1, filename)
        print(wav_path)
        result = ans(wav_path,output_path=outputPath)


# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='damo/speech_frcrn_ans_cirm_16k')
# result = ans(
#     '/home/liu70kg/D512G/SEARCH/audio/sheyang/answer_video_7878_52_1_4.wav',
#     output_path='./processed.wav')


