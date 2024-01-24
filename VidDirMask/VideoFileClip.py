from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip


def replace_audio(video_path_a, audio_path_b, output_video_path):
    # 加载视频A
    video_clip_a = VideoFileClip(video_path_a)

    # 加载音频B
    audio_clip_b = AudioFileClip(audio_path_b)

    # 提取视频A的音频
    video_audio_a = video_clip_a.audio

    # 用音频B替换视频A的音频
    video_clip_b = video_clip_a.set_audio(audio_clip_b)

    # 合成新的视频B
    video_b = CompositeVideoClip([video_clip_b])

    # 保存输出视频
    video_b.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # 关闭视频和音频文件
    video_clip_a.close()
    video_clip_b.close()
    audio_clip_b.close()


def extract_and_save_audio(video_path, output_audio_path):
    # 加载视频A
    video_clip = VideoFileClip(video_path)

    # 提取音频
    audio_clip = video_clip.audio

    # 保存音频
    audio_clip.write_audiofile(output_audio_path, codec='mp3')  # 可以选择其他音频格式

    # 关闭视频和音频文件
    video_clip.close()
    audio_clip.close()

