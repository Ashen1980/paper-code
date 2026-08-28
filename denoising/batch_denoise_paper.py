"""
batch_denoise_paper.py — 批量降噪：对 001_paper015 下所有原始 wav 降噪

输入:  001_paper015/{AS,MR,MS,MVP,N}/*.wav
输出:  001_paper015/{AS_denoised,MR_denoised,MS_denoised,MVP_denoised,N_denoised}/*.wav
"""

import os, sys, time
from math import gcd
from typing import Optional, Tuple, List
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
from scipy.ndimage import grey_opening, grey_closing
import pywt

BASE_DIR = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\001_paper015"
TARGET_SR = 8000
OUTPUT_PEAK = 0.98

LOW_HZ, HIGH_HZ = 25.0, 500.0
WAVELET_LEVEL = 5
BASE_THRESHOLD_SCALE = 0.38
DETAIL_THRESHOLD_WEIGHTS = [1.00, 0.85, 0.65, 0.45, 0.30]
ACTIVE_THR_MIN, ACTIVE_THR_MAX = 0.65, 1.45
ACTIVE_BLEND_MIN, ACTIVE_BLEND_MAX = 0.02, 0.30
USE_NOISE_GATE = True
NOISE_GATE_STRENGTH, NOISE_GATE_FLOOR_RATIO = 0.24, 1.55
USE_CLUSTER_SUPPRESS = True
CLUSTER_SUPPRESS_STRENGTH = 0.16
USE_LOWFREQ_FLOOR_SUPPRESS = True
LOWFREQ_FLOOR_STRENGTH = 2.5
LOWFREQ_FLOOR_BAND = (10.0, 60.0)
MASK_LOW_HZ = 25.0
BASELINE_WIN_SEC = 0.15

CATEGORIES = ["AS", "MR", "MS", "MVP", "N"]

# ── 算法 (从 denoising_v2.py) ──
def load_and_resample(path, target_sr=None):
    sr_orig, x = wavfile.read(path)
    if x.dtype == np.int16: y = x.astype(np.float64)/32768.0
    elif x.dtype == np.int32: y = x.astype(np.float64)/2147483648.0
    elif np.issubdtype(x.dtype, np.floating): y = x.astype(np.float64)
    else: raise TypeError(f"unsupported dtype: {x.dtype}")
    if y.ndim == 2: y = np.mean(y, axis=1)
    if target_sr is not None and sr_orig != target_sr:
        from scipy.signal import resample_poly
        g = gcd(int(target_sr), int(sr_orig))
        up, down = int(target_sr)//g, int(sr_orig)//g
        y = resample_poly(y, up, down); sr_orig = int(target_sr)
    return sr_orig, y

def save_audio(path, sr, x):
    y = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(y))+1e-12
    y = y/peak*OUTPUT_PEAK
    y16 = np.clip(y*32767.0, -32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wavfile.write(path, sr, y16)

def butter_bandpass_zero_phase(x, sr, low_hz, high_hz, order=4):
    nyq = sr/2.0
    low = max(0.5, low_hz)/nyq; high = min(high_hz, nyq-1.0)/nyq
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, x).astype(np.float64)

def moving_average(x, win):
    win = max(1, int(win))
    return np.convolve(x, np.ones(win)/win, mode="same")

def compute_activity_mask(x_band, sr):
    analytic = sig.hilbert(x_band)
    env = np.abs(np.asarray(analytic))
    env = moving_average(env, int(0.03*sr))
    p10, p95 = np.percentile(env, 10), np.percentile(env, 95)
    denom = max(p95-p10, 1e-8)
    act = np.clip((env-p10)/denom, 0.0, 1.0)
    act = moving_average(act, int(0.05*sr))
    return np.clip(act, 0.0, 1.0)

def _pad_even(x):
    if len(x)%2==0: return x, 0
    return np.pad(x, (0, 1), mode="edge"), 1

def haar_wavedec(x, level=5):
    a = x.astype(np.float64).copy()
    details, pads = [], []
    inv_sqrt2 = 1.0/np.sqrt(2.0)
    for _ in range(level):
        if len(a)<4: break
        a_pad, pad = _pad_even(a)
        even, odd = a_pad[0::2], a_pad[1::2]
        approx = (even+odd)*inv_sqrt2; detail = (even-odd)*inv_sqrt2
        details.append(detail); pads.append(pad); a = approx
    return a, details, pads

def haar_waverec(approx, details, pads):
    a = approx.astype(np.float64).copy()
    inv_sqrt2 = 1.0/np.sqrt(2.0)
    for detail, pad in zip(details[::-1], pads[::-1]):
        even = (a+detail)*inv_sqrt2; odd = (a-detail)*inv_sqrt2
        out = np.empty(even.size+odd.size, dtype=np.float64)
        out[0::2]=even; out[1::2]=odd
        if pad==1: out=out[:-1]
        a=out
    return a

def downsample_profile(profile, target_len):
    idx = np.linspace(0, len(profile), target_len+1, dtype=int)
    out = np.zeros(target_len, dtype=np.float64)
    for i in range(target_len):
        seg = profile[idx[i]:idx[i+1]]
        out[i] = np.mean(seg) if len(seg)>0 else 0.0
    return out

def wavelet_denoise_adaptive(x, activity, level=5, base_scale=0.38, detail_weights=None):
    if detail_weights is None: detail_weights = [1.00, 0.85, 0.60, 0.38, 0.24]
    approx, details, pads = haar_wavedec(x, level=level)
    proc_details = []
    sigma = np.median(np.abs(details[0]))/0.6745+1e-12
    for i, detail in enumerate(details):
        weight = detail_weights[min(i, len(detail_weights)-1)]
        thr_global = base_scale*sigma*np.sqrt(2.0*np.log(len(detail)+1.0))*weight
        prof = downsample_profile(activity, len(detail))
        thr_scale = ACTIVE_THR_MAX-(ACTIVE_THR_MAX-ACTIVE_THR_MIN)*prof
        thr_arr = thr_global*thr_scale
        d = np.sign(detail)*np.maximum(np.abs(detail)-thr_arr, 0.0)
        proc_details.append(d)
    y = haar_waverec(approx, proc_details, pads)
    return y[:len(x)]

def noise_gate_inactive_only(x, sr, activity, strength, floor_ratio):
    env = np.abs(np.asarray(sig.hilbert(x)))
    env_s = moving_average(env, int(0.02*sr))
    floor = np.percentile(env_s, 20); threshold = floor*floor_ratio
    non_active = 1.0-np.clip(activity, 0.0, 1.0)
    excess = np.maximum(threshold-env_s, 0.0)/(threshold+1e-12)
    gain = 1.0-strength*non_active*excess
    return x*np.clip(gain, 0.78, 1.0)

def suppress_lowfreq_floor(x, sr, activity, band, strength):
    nyq = sr/2.0
    b, a = sig.butter(2, [max(1.0, band[0])/nyq, min(band[1], nyq-1.0)/nyq], btype="band")
    x_low = sig.filtfilt(b, a, x)
    env_s = moving_average(np.abs(np.asarray(sig.hilbert(x_low))), int(0.06*sr))
    non_active = 1.0-np.clip(activity, 0.0, 1.0)
    floor = np.percentile(env_s, 45)
    excess = np.maximum(env_s-floor, 0.0)
    gain = np.clip(1.0/(1.0+strength*non_active*(excess/(floor+1e-8))), 0.15, 1.0)
    return x-x_low+(x_low*gain)

def transient_cluster_suppress(x, sr, band, strength):
    nyq = sr/2.0
    b, a = sig.butter(2, [band[0]/nyq, min(band[1], nyq-1.0)/nyq], btype="band")
    filtered = sig.filtfilt(b, a, x)
    env_s = moving_average(np.abs(np.asarray(sig.hilbert(filtered))), int(0.025*sr))
    limit = np.percentile(env_s, 92)
    gain = np.ones_like(env_s); over = env_s>limit
    gain[over] = 1.0/(1.0+strength*(env_s[over]/(limit+1e-8)-1.0))
    return x*gain

def remove_baseline_morphology(x, sr, win_sec=0.15):
    win_len = int(win_sec*sr)
    baseline = grey_opening(x, size=win_len)
    baseline = grey_closing(baseline, size=win_len)
    return x-baseline

def denoise_heart_sound(x, sr):
    x_stable = remove_baseline_morphology(x, sr, win_sec=BASELINE_WIN_SEC)
    x_band = butter_bandpass_zero_phase(x_stable, sr, LOW_HZ, HIGH_HZ, order=4)
    x_for_mask = butter_bandpass_zero_phase(x_stable, sr, MASK_LOW_HZ, HIGH_HZ, order=2)
    activity = compute_activity_mask(x_for_mask, sr)
    y = wavelet_denoise_adaptive(x_stable, activity, level=WAVELET_LEVEL,
        base_scale=BASE_THRESHOLD_SCALE, detail_weights=DETAIL_THRESHOLD_WEIGHTS)
    if USE_NOISE_GATE: y = noise_gate_inactive_only(y, sr, activity, NOISE_GATE_STRENGTH, NOISE_GATE_FLOOR_RATIO)
    if USE_LOWFREQ_FLOOR_SUPPRESS: y = suppress_lowfreq_floor(y, sr, activity, LOWFREQ_FLOOR_BAND, LOWFREQ_FLOOR_STRENGTH)
    if USE_CLUSTER_SUPPRESS: y = transient_cluster_suppress(y, sr, band=(25.0, HIGH_HZ), strength=CLUSTER_SUPPRESS_STRENGTH)
    alpha = ACTIVE_BLEND_MIN+(ACTIVE_BLEND_MAX-ACTIVE_BLEND_MIN)*np.clip(activity, 0.0, 1.0)
    y = (1.0-alpha)*y+alpha*x_band
    return butter_bandpass_zero_phase(y, sr, LOW_HZ, HIGH_HZ, order=2)

# ── 批量处理 ──
def main():
    t0 = time.time()
    total_all = 0
    for cat in CATEGORIES:
        in_dir = os.path.join(BASE_DIR, cat)
        out_dir = os.path.join(BASE_DIR, f"{cat}_denoised")
        os.makedirs(out_dir, exist_ok=True)
        wavs = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".wav")])
        total_all += len(wavs)
        print(f"\n{cat}: {len(wavs)} files → {cat}_denoised/")
        for i, fname in enumerate(wavs, 1):
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(out_path):
                if i % 50 == 0: print(f"  [{i:3d}/{len(wavs)}] (skip)")
                continue
            try:
                sr, x = load_and_resample(in_path, TARGET_SR)
                x_dn = denoise_heart_sound(x, sr)
                save_audio(out_path, sr, x_dn)
                if i % 50 == 0:
                    el = time.time()-t0
                    print(f"  [{i:3d}/{len(wavs)}]  {el:.0f}s elapsed")
            except Exception as e:
                print(f"  [{i:3d}/{len(wavs)}] ERR {fname}: {e}")
    print(f"\nDone! {total_all} files, {time.time()-t0:.0f}s total")

if __name__ == "__main__":
    main()
