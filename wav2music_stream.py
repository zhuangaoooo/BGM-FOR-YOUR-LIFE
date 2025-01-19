import pygame
import os
import time

pygame.mixer.init()  # 初始化pygame的混音器模块

def play_music_with_smooth_crossfade(directory, crossfade_time=2):
    # 获取目录中所有wav文件，并按修改时间排序
    wav_files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
    if not wav_files:
        print("No WAV files found in the directory.")
        return

    # 找到最新的wav文件
    latest_wav = max(wav_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    current_wav = latest_wav
    current_file_path = os.path.join(directory, current_wav)

    # 播放最新的wav文件
    print(f"Playing WAV file: {current_wav}")
    pygame.mixer.music.load(current_file_path)
    pygame.mixer.music.play(-1)  # 循环播放

    while True:  # 循环检查是否有新文件
        wav_files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
        latest_wav = max(wav_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))

        if latest_wav != current_wav:  # 如果有新的音乐文件
            current_wav = latest_wav
            next_file_path = os.path.join(directory, current_wav)
            print(f"New WAV file detected: {current_wav}")
            # 交叉淡入淡出
            smooth_crossfade(current_file_path, next_file_path, crossfade_time)
            current_file_path = next_file_path
        else:  # 如果没有新的音乐文件，继续播放当前音乐
            continue

        time.sleep(1)  # 每1秒检查一次新文件

def smooth_crossfade(old_file_path, new_file_path, crossfade_time):
    # 加载新音乐并设置初始音量为0
    new_track = pygame.mixer.Sound(new_file_path)
    new_track.set_volume(0.0)
    new_track.play(-1)  # 循环播放

    # 获取当前音乐的音量
    current_volume = pygame.mixer.music.get_volume()

    # 交叉淡出旧音乐和淡入新音乐
    increment = 1.0 / (crossfade_time * 10)  # 每次循环增加的音量比例
    for i in range(int(crossfade_time * 1000)):  # 持续时间为crossfade_time毫秒
        pygame.time.wait(1)  # 等待1毫秒
        # 降低旧音乐的音量
        if current_volume > 0:
            current_volume -= increment
            pygame.mixer.music.set_volume(max(0.0, current_volume))
        # 增加新音乐的音量
        new_track.set_volume(min(1.0, new_track.get_volume() + increment))

    # 停止旧音乐
    pygame.mixer.music.stop()
    # 确保新音乐继续播放
    new_track.play(-1)

directory = 'C:\\Users\\张博伦\\Desktop\\大三上\\ai\\wav'
play_music_with_smooth_crossfade(directory)