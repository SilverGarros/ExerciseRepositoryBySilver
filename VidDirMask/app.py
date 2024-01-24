# backend.py
import os

import gradio as gr
from gradio import Interface
import VideoFileClip
from function import get_filename_without_extension
from whispertest import TTSByWhisper, overlay_silence, replace_silence_with_beep


def delect_cache(directory):
    # 删除目录下的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def process_video(input_video_path):
    # 视频完整的处理过程
    os.makedirs('cache', exist_ok=True)
    os.makedirs('save', exist_ok=True)

    video_name = get_filename_without_extension(input_video_path)
    audio_cache = "cache/" + video_name + "_cache.mp3"
    VideoFileClip.extract_and_save_audio(input_video_path, audio_cache)
    # 进行Whisper(默认medium模型)文本分析，并获取所有含有脏话预计的时间段的起止时间戳
    print("TTSByWhisper"+os.getcwd())
    time_intervals = TTSByWhisper(audio_cache)
    # 调用函数完成时间段消音
    audio_cache_after_silence = overlay_silence(audio_cache, time_intervals)
    # 将消声的部分替换为Bee声
    beep_sound_path = "audio/beep.mp3"
    audio_cache_after_replece_silence_with_beep = replace_silence_with_beep(audio_cache_after_silence, time_intervals,
                                                                            beep_sound_path)
    # 将原视频的音频替换为经过脏话过滤的音频并输出返回
    output_video_path = "save/" + video_name + "_Filtered.mp4"
    print("正在进行过滤后的音频替换")
    VideoFileClip.replace_audio(input_video_path, audio_cache_after_replece_silence_with_beep, output_video_path)
    print("脏话过滤已经完成")
    # 删除中间缓存文件
    print("正在删除中间缓存文件。。。")
    delect_cache('cache')
    print("缓存文件删除完毕，脏话过滤后的视频已经保存为"+"save/" + video_name + "_Filtered.mp4")
    return output_video_path


iface = gr.Interface(fn=process_video, inputs="file", outputs="file")
iface.launch(share=True)
