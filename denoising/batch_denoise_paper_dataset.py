"""
batch_denoise_paper_dataset.py — 用论文 II-B 降噪方法(db6)重新降噪 paper 数据集。

输入:  001_paper015/{AS,MR,MS,MVP,N}/*.wav   (8000 Hz, 单声道)
输出:  001_paper015/{AS,MR,MS,MVP,N}_denoised/*.wav  (覆盖旧 Haar 结果)

论文方法顺序(与 batch_denoise_paper_method.py 完全一致):
  (ii) DC 去除
  (iii) 幅值削波(99.5 分位)
  (iv) 4 阶 Butterworth 带通 20-500 Hz 零相位
  (v) 5 层 db6 小波 + BayesShrink 自适应软阈值
  (vi) Hilbert 包络增强
  (vii) 65/35 混音 + RMS 归一化
"""
import os
from math import gcd

import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import pywt

BASE = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\001_paper015"
CATEGORIES = ["AS", "MR", "MS", "MVP", "N"]

TARGET_SR = 8000
LOW_HZ, HIGH_HZ = 20.0, 500.0
BP_ORDER = 4
WAVELET = "db6"
WAVELET_LEVEL = 5
BAYES_BETA = 1.0
CLIP_PCT = 99.5
ENV_SMOOTH_SEC = 0.04
ENV_MAX_GAIN = 2.0
BLEND_DEN, BLEND_ORG = 0.65, 0.35


def load_and_resample(path, target_sr):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64)
    else:
        raise TypeError(f"unsupported dtype: {x.dtype}")
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if target_sr is not None and sr != target_sr:
        g = gcd(int(target_sr), int(sr))
        y = sig.resample_poly(y, int(target_sr) // g, int(sr) // g)
        sr = int(target_sr)
    return sr, y


def save_audio(path, sr, x):
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)
    wavfile.write(path, sr, y16)


def moving_average(x, win):
    win = max(1, int(win))
    return np.convolve(x, np.ones(win) / win, mode="same")


def butter_bandpass_zero_phase(x, sr, low, high, order=4):
    nyq = sr / 2.0
    b, a = sig.butter(order, [max(0.5, low) / nyq, min(high, nyq - 1.0) / nyq], btype="band")
    return sig.filtfilt(b, a, x).astype(np.float64)


def bayes_shrink(x, wavelet, level, beta):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    details = coeffs[1:]
    sigma = np.median(np.abs(details[-1])) / 0.6745
    new_details = []
    for d in details:
        var_d = np.var(d)
        sigma_x = np.sqrt(max(var_d - sigma * sigma, 1e-12))
        T = beta * sigma * sigma / sigma_x
        new_details.append(pywt.threshold(d, T, mode="soft"))
    return pywt.waverec([coeffs[0]] + new_details, wavelet)


def hilbert_envelope_enhance(x, sr, smooth_sec, max_gain):
    env = np.abs(sig.hilbert(x))
    env_s = moving_average(env, int(smooth_sec * sr))
    ref = np.median(env_s) + 1e-12
    gain = np.clip(env_s / ref, 0.0, max_gain)
    return x * gain


def denoise_paper_method(x, sr):
    x = x - np.mean(x)                                            # DC
    clip_at = np.percentile(np.abs(x), CLIP_PCT)                  # 削波
    x = np.clip(x, -clip_at, clip_at)
    x_bp = butter_bandpass_zero_phase(x, sr, LOW_HZ, HIGH_HZ, BP_ORDER)
    x_dwt = bayes_shrink(x_bp, WAVELET, WAVELET_LEVEL, BAYES_BETA)[:len(x_bp)]
    x_den = hilbert_envelope_enhance(x_dwt, sr, ENV_SMOOTH_SEC, ENV_MAX_GAIN)
    x_mix = BLEND_DEN * x_den + BLEND_ORG * x_bp                  # 65/35 混音
    rms_orig = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_mix = np.sqrt(np.mean(x_mix ** 2)) + 1e-12
    return x_mix * (rms_orig / rms_mix)                           # RMS 归一化


def main():
    total = 0
    ok = 0
    for cat in CATEGORIES:
        src_dir = os.path.join(BASE, cat)
        dst_dir = os.path.join(BASE, f"{cat}_denoised")
        os.makedirs(dst_dir, exist_ok=True)
        wavs = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".wav"))
        for i, fn in enumerate(wavs, 1):
            total += 1
            in_path = os.path.join(src_dir, fn)
            out_path = os.path.join(dst_dir, fn)
            try:
                sr, x = load_and_resample(in_path, TARGET_SR)
                x_den = denoise_paper_method(x, sr)
                save_audio(out_path, sr, x_den)
                ok += 1
            except Exception as e:
                print(f"[ERR] {cat}/{fn}: {e}", flush=True)
            if i % 100 == 0 or i == len(wavs):
                print(f"[{cat}] {i}/{len(wavs)}", flush=True)
    print(f"FINISHED: {ok}/{total} files denoised", flush=True)


if __name__ == "__main__":
    main()