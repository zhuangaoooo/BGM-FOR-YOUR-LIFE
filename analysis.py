import numpy as np
import librosa
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class AudioQualityEvaluator:

    def __init__(self, audio_folder):
        self.audio_folder = audio_folder

    # 计算信噪比（SNR）
    def calculate_snr(self, audio_file):
        y, sr = librosa.load(audio_file)
        # 分离谐波和打击乐
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 计算信号和噪声的功率
        signal_power = np.mean(y_harmonic**2)
        noise_power = np.mean(y_percussive**2)

        # 计算SNR
        if noise_power == 0:  # 避免除以零的情况
            snr = float('inf')
        else:
            snr = 10 * np.log10(signal_power / noise_power)
        
        return snr

    # 根据SNR评估音频质量
    def classify_audio_quality(self, snr):
        if snr > 20:
            return "好"
        elif 10 <= snr <= 20:
            return "中"
        else:
            return "差"

    # 计算MFCC特征
    def extract_mfcc(self, audio_file, n_mfcc=13):
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # 计算MFCC的均值，将其转化为一维特征向量

    # 估计音频的BPM（节拍数）
    def estimate_bpm(self, audio_file):
        y, sr = librosa.load(audio_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return tempo  # 返回BPM（节拍数）

    # 根据BPM估计情感
    def classify_emotion_by_bpm(self, bpm):
        if 90 <= bpm <= 208:
            return "欢快，紧张"
        elif 80 <= bpm < 90:
            return "平缓，正常"
        elif bpm < 80:
            return "缓慢，温情，忧伤"
        else:
            return "无法分类"

    # 计算两个音频文件的DTW距离
    def compute_dtw(self, mfcc1, mfcc2):
        # 确保输入是二维数组
        mfcc1 = np.reshape(mfcc1, (-1, 1))  # 转换为(N, 1)
        mfcc2 = np.reshape(mfcc2, (-1, 1))  # 转换为(N, 1)

        # 使用DTW计算两个一维特征向量之间的距离
        distance, _ = fastdtw(mfcc1, mfcc2, dist=euclidean)
        return distance

    # 主功能：遍历文件夹并计算音频文件之间的相似度
    def calculate_similarity_matrix(self):
        # 存储音频文件的名称和对应的MFCC特征
        music_files = []
        mfcc_features = []
        bpm_values = []
        emotions = []
        snr_values = []
        quality_values = []

        # 遍历音频文件夹中的所有音频文件
        for file_name in os.listdir(self.audio_folder):
            if file_name.endswith(('.wav', '.mp3', '.flac')):  # 支持的音频文件格式
                file_path = os.path.join(self.audio_folder, file_name)
                try:
                    # 提取当前音频文件的MFCC
                    music_mfcc = self.extract_mfcc(file_path)
                    bpm = self.estimate_bpm(file_path)
                    emotion = self.classify_emotion_by_bpm(bpm)
                    snr = self.calculate_snr(file_path)
                    quality = self.classify_audio_quality(snr)

                    music_files.append(file_name)
                    mfcc_features.append(music_mfcc)
                    bpm_values.append(bpm)
                    emotions.append(emotion)
                    snr_values.append(snr)
                    quality_values.append(quality)

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        # 计算相似度矩阵
        num_files = len(mfcc_features)
        similarity_matrix = np.zeros((num_files, num_files))

        for i in range(num_files):
            for j in range(i, num_files):
                # 计算两个文件之间的DTW距离
                dist = self.compute_dtw(mfcc_features[i], mfcc_features[j])
                similarity_matrix[i][j] = dist
                similarity_matrix[j][i] = dist  # 相似度矩阵是对称的

        return music_files, similarity_matrix, bpm_values, emotions, snr_values, quality_values

    # 可视化相似度矩阵
    def plot_similarity_matrix(self, music_files, similarity_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, xticklabels=music_files, yticklabels=music_files, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title("音频文件相似度矩阵")
        plt.xlabel("音频文件")
        plt.ylabel("音频文件")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # 显示音频的BPM、情感、SNR和音频质量
    def plot_bpm_emotions_snr_and_quality(self, music_files, bpm_values, emotions, snr_values, quality_values):
        for file_name, bpm, emotion, snr, quality in zip(music_files, bpm_values, emotions, snr_values, quality_values):
            if isinstance(bpm, np.ndarray):
                bpm = bpm[0]  # 如果bpm是数组，取第一个值
            print(f"文件: {file_name} | BPM: {bpm:.2f} | 情感: {emotion} | SNR: {snr:.2f} dB | 质量: {quality}")

# 使用示例
audio_folder = r"D:\ISE3309\wav"  # 音频文件夹路径
evaluator = AudioQualityEvaluator(audio_folder)

# 计算相似度矩阵
music_files, similarity_matrix, bpm_values, emotions, snr_values, quality_values = evaluator.calculate_similarity_matrix()

# 打印BPM、情感、SNR和音频质量
evaluator.plot_bpm_emotions_snr_and_quality(music_files, bpm_values, emotions, snr_values, quality_values)

# 可视化相似度矩阵
evaluator.plot_similarity_matrix(music_files, similarity_matrix)
