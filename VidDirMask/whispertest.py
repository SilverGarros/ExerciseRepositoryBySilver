import os

import whisper
from pydub import AudioSegment

from function import get_filename_without_extension


def TTSByWhisper(audio_cache):
    print(os.getcwd())
    whisper_model = whisper.load_model("medium")
    result = whisper_model.transcribe(audio_cache)
    print(",".join([i["text"] for i in result["segments"] if i is not None]))
    print(result)
    time_intervals = []

    with open('model/keywords.txt', 'r', encoding='utf-8') as file:
        keywords = [line.strip() for line in file]
        print(keywords)

    for segment in result['segments']:
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']

        if any(keyword in text for keyword in keywords):
            print(f"Found keywords in segment {segment['id']}:")
            print(f"Start time: {start_time}, End time: {end_time}")
            time_intervals.append((start_time, end_time))

    print(time_intervals)
    return time_intervals


def adjust_volume_in_range(audio, start_ms, end_ms, target_db):
    # 确保时间范围在有效范围内
    start_ms = max(0, start_ms)
    end_ms = min(end_ms, len(audio))

    # 截取特定时间范围内的音频
    target_range = audio[start_ms:end_ms]

    # 计算增益（dB）
    current_db = target_range.dBFS
    gain = target_db - current_db

    # 调整音量
    target_range = target_range + gain

    # 将调整后的音频替换回原始音频中的相应时间范围
    audio = audio[:start_ms] + target_range + audio[end_ms:]
    return audio


def overlay_silence(audio_file_path, time_intervals):
    # 加载音频文件
    audio_file = AudioSegment.from_file(audio_file_path)

    # 将时间间隔内的音频进行静音覆盖
    for start, end in time_intervals:
        audio_file = adjust_volume_in_range(audio_file, start * 1000, end * 1000, -120.0)  # 使用-120 dB 表示完全静音

    # 输出处理后的音频
    output_path = "cache/" + get_filename_without_extension(audio_file_path) + "_silence.mp3"
    print("将脏话消声后的音频保存为" + output_path)
    audio_file.export(output_path, format="mp3")
    return output_path


def replace_with_beep(audio, start_ms, end_ms, beep_path):
    # 确保时间范围在有效范围内
    start_ms = max(0, start_ms)
    end_ms = min(end_ms, len(audio))

    # 截取特定时间范围内的音频
    target_range = audio[start_ms:end_ms]

    # 载入“哔”的音效文件
    beep_sound = AudioSegment.from_file(beep_path)

    # 调整音效文件的长度以匹配目标范围
    beep_sound = beep_sound[:len(target_range)]

    # 将“哔”的音效替换到原始音频中的相应时间范围
    audio = audio[:start_ms] + beep_sound + audio[end_ms:]
    return audio


def replace_silence_with_beep(audio_file_path, time_intervals, beep_path):
    # 加载音频文件
    audio_file = AudioSegment.from_file(audio_file_path)

    # 将时间间隔内的静音区域替换为“哔”的音效
    print(time_intervals)
    for time_interval in time_intervals:
        print(time_interval)
        if len(time_interval) != 2:
            print(f"Error: Invalid time interval format: {time_interval}")
            continue
        start, end = time_interval
        audio_file = replace_with_beep(audio_file, start * 1000, end * 1000, beep_path)

    # 输出处理后的音频
    output_path = "cache/" + get_filename_without_extension(audio_file_path) + "_replace_silence_with_beep.mp3"
    audio_file.export(output_path, format="mp3")
    print("脏话消声已经完成！")
    return output_path
