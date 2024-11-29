from whisper import whisper
import opencc

# 加载Whisper模型
model = whisper.load_model("small")  # 可选择 "tiny", "base", "small", "medium", "large" 等模型

# 加载音频文件
audio_file_path = '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_read/sheyang/read_video_9038_52_1_39.wav'  # 替换为您的音频文件路径
audio = whisper.load_audio(audio_file_path)
audio = whisper.pad_or_trim(audio)

# 进行语音识别
result = model.transcribe(audio_file_path, word_timestamps=True, language='zh')
# 输出识别结果和时间戳
print("识别文本:", result['text'])
# 繁体转简体
converter = opencc.OpenCC('t2s')  # 't2s.json'表示从繁体转到简体
simplified_text = converter.convert(result['text'])
print("简体文本:", simplified_text)
for word_info in result['segments']:
    print(f"单词: {word_info['text']}, 开始时间: {word_info['start']}, 结束时间: {word_info['end']}")