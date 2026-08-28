"""
denoising_v2.py — 心音小波降噪 V2 (升级版 + 重采样 + 可视化)

升级点 (相对于 denoising.py):
  形态学基线消除 → 25-240Hz 巴特沃斯带通 → 5层Haar小波自适应阈值降噪
  → 噪声门 → 低频地板抑制(10-60Hz) → 瞬态簇抑制(25-240Hz)
  → active区原信号回混 → 最终25-240Hz带通收尾

关键改进:
  - activity掩码脱钩: 25Hz高通信号计算activity, 避免<25Hz低频伪影欺骗检测器
  - 形态学基线消除: grey_opening+grey_closing 消除听诊器摩擦等极低频伪影
  - 自适应阈值: 每层小波系数根据activity动态调整阈值强度
  - 支持重采样: TARGET_SR 设为 8000 可统一输出采样率

输入:  修改 INPUT_DIR + SAMPLE_NAME 或直接修改 INPUT_WAV
输出:  降噪后 wav (OUTPUT_DIR) + 对比图 (OUTPUT_PLOT_DIR)
"""

import os
from math import gcd
from typing import Optional, Tuple, List
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
from scipy.ndimage import grey_opening, grey_closing
import matplotlib.pyplot as plt
import pywt

# =========================
# 1) 顶部配置区 — 修改这里的路径
# =========================
INPUT_DIR = r"C:\Users\26287\Desktop\Work\Feature Extraction\人工混合数据集\synth_mix"
SAMPLE_NAME = "synth_000100_mix.wav"
INPUT_WAV = os.path.join(INPUT_DIR, SAMPLE_NAME)

OUTPUT_DIR = r"C:\Users\26287\Desktop\Work\Feature Extraction\人工混合数据集\synth_denoised_v2"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, os.path.splitext(SAMPLE_NAME)[0] + "_denoised.wav")
OUTPUT_PLOT_DIR = r"C:\Users\26287\Desktop\Work\Feature Extraction\compare_results\v2_test"

# --- 重采样参数 ---
TARGET_SR = 8000  # 设为 None 保留原始采样率; 8000 统一到标准采样率

# --- 算法参数 (新版优化) ---
LOW_HZ = 25.0
HIGH_HZ = 500.0

WAVELET_LEVEL = 5
BASE_THRESHOLD_SCALE = 0.38
DETAIL_THRESHOLD_WEIGHTS = [1.00, 0.85, 0.65, 0.45, 0.30]

ACTIVE_THR_MIN = 0.65
ACTIVE_THR_MAX = 1.45
ACTIVE_BLEND_MIN = 0.02
ACTIVE_BLEND_MAX = 0.12

USE_NOISE_GATE = True
NOISE_GATE_STRENGTH = 0.24
NOISE_GATE_FLOOR_RATIO = 1.55

USE_CLUSTER_SUPPRESS = True
CLUSTER_SUPPRESS_STRENGTH = 0.16

USE_LOWFREQ_FLOOR_SUPPRESS = True
LOWFREQ_FLOOR_STRENGTH = 2.5
LOWFREQ_FLOOR_BAND = (10.0, 60.0)

OUTPUT_PEAK = 0.98

# Activity 掩码脱钩用高通截止
MASK_LOW_HZ = 25.0

# 形态学基线消除窗宽
BASELINE_WIN_SEC = 0.15

# --- 画图参数 ---
PLOT_WAVELET_NAME = "db6"
MAX_SECONDS_TO_SHOW = 8.0
FIG_DPI = 150

# =========================
# 2) 音频 IO + 重采样
# =========================
def load_and_resample(path: str, target_sr: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """加载 wav 并可选重采样到 target_sr"""
    sr_orig, x = wavfile.read(path)
    if x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64)
    else:
        raise TypeError(f"不支持的音频类型: {x.dtype}")
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    if target_sr is not None and sr_orig != target_sr:
        from scipy.signal import resample_poly
        g = gcd(int(target_sr), int(sr_orig))
        up = int(target_sr) // g
        down = int(sr_orig) // g
        y = resample_poly(y, up, down)
        sr_orig = int(target_sr)
        print(f"  重采样: {sr_orig} -> {target_sr} Hz")

    return sr_orig, y


def save_audio(path: str, sr: int, x: np.ndarray) -> None:
    y = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak * OUTPUT_PEAK
    y16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wavfile.write(path, sr, y16)


# =========================
# 3) 核心降噪算法 (新版)
# =========================
def butter_bandpass_zero_phase(x: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    low = max(0.5, low_hz) / nyq
    high = min(high_hz, nyq - 1.0) / nyq
    b, a = sig.butter(order, [low, high], btype="band")  # type: ignore[assignment]
    return sig.filtfilt(b, a, x).astype(np.float64)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, kernel, mode="same")


def compute_activity_mask(x_band: np.ndarray, sr: int) -> np.ndarray:
    analytic = sig.hilbert(x_band)  # type: ignore[assignment]
    env = np.abs(np.asarray(analytic))
    env = moving_average(env, int(0.03 * sr))
    p10 = np.percentile(env, 10)
    p95 = np.percentile(env, 95)
    denom = max(p95 - p10, 1e-8)
    act = np.clip((env - p10) / denom, 0.0, 1.0)
    act = moving_average(act, int(0.05 * sr))
    return np.clip(act, 0.0, 1.0)


def _pad_even(x: np.ndarray) -> Tuple[np.ndarray, int]:
    if len(x) % 2 == 0:
        return x, 0
    return np.pad(x, (0, 1), mode="edge"), 1


def haar_wavedec(x: np.ndarray, level: int = 5) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    a = x.astype(np.float64).copy()
    details, pads = [], []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for _ in range(level):
        if len(a) < 4:
            break
        a_pad, pad = _pad_even(a)
        even, odd = a_pad[0::2], a_pad[1::2]
        approx = (even + odd) * inv_sqrt2
        detail = (even - odd) * inv_sqrt2
        details.append(detail)
        pads.append(pad)
        a = approx
    return a, details, pads


def haar_waverec(approx: np.ndarray, details: List[np.ndarray], pads: List[int]) -> np.ndarray:
    a = approx.astype(np.float64).copy()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for detail, pad in zip(details[::-1], pads[::-1]):
        even = (a + detail) * inv_sqrt2
        odd = (a - detail) * inv_sqrt2
        out = np.empty(even.size + odd.size, dtype=np.float64)
        out[0::2] = even
        out[1::2] = odd
        if pad == 1:
            out = out[:-1]
        a = out
    return a


def downsample_profile(profile: np.ndarray, target_len: int) -> np.ndarray:
    idx = np.linspace(0, len(profile), target_len + 1, dtype=int)
    out = np.zeros(target_len, dtype=np.float64)
    for i in range(target_len):
        seg = profile[idx[i]:idx[i + 1]]
        out[i] = np.mean(seg) if len(seg) > 0 else 0.0
    return out


def wavelet_denoise_adaptive(
    x: np.ndarray, activity: np.ndarray,
    level: int = 5, base_scale: float = 0.38,
    detail_weights: Optional[List[float]] = None
) -> np.ndarray:
    if detail_weights is None:
        detail_weights = [1.00, 0.85, 0.60, 0.38, 0.24]
    approx, details, pads = haar_wavedec(x, level=level)
    proc_details = []
    sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
    for i, detail in enumerate(details):
        weight = detail_weights[min(i, len(detail_weights) - 1)]
        thr_global = base_scale * sigma * np.sqrt(2.0 * np.log(len(detail) + 1.0)) * weight
        prof = downsample_profile(activity, len(detail))
        thr_scale = ACTIVE_THR_MAX - (ACTIVE_THR_MAX - ACTIVE_THR_MIN) * prof
        thr_arr = thr_global * thr_scale
        d = np.sign(detail) * np.maximum(np.abs(detail) - thr_arr, 0.0)
        proc_details.append(d)
    y = haar_waverec(approx, proc_details, pads)
    return y[:len(x)]


def noise_gate_inactive_only(
    x: np.ndarray, sr: int, activity: np.ndarray,
    strength: float, floor_ratio: float
) -> np.ndarray:
    env = np.abs(np.asarray(sig.hilbert(x)))  # type: ignore[arg-type]
    env_s = moving_average(env, int(0.02 * sr))
    floor = np.percentile(env_s, 20)
    threshold = floor * floor_ratio
    non_active = 1.0 - np.clip(activity, 0.0, 1.0)
    excess = np.maximum(threshold - env_s, 0.0) / (threshold + 1e-12)
    gain = 1.0 - strength * non_active * excess
    return x * np.clip(gain, 0.78, 1.0)


def suppress_lowfreq_floor(
    x: np.ndarray, sr: int, activity: np.ndarray,
    band: Tuple[float, float], strength: float
) -> np.ndarray:
    nyq = sr / 2.0
    b, a = sig.butter(2, [max(1.0, band[0]) / nyq, min(band[1], nyq - 1.0) / nyq], btype="band")  # type: ignore[assignment]
    x_low = sig.filtfilt(b, a, x)
    env_s = moving_average(np.abs(np.asarray(sig.hilbert(x_low))), int(0.06 * sr))  # type: ignore[arg-type]
    non_active = 1.0 - np.clip(activity, 0.0, 1.0)
    floor = np.percentile(env_s, 45)
    excess = np.maximum(env_s - floor, 0.0)
    gain = np.clip(1.0 / (1.0 + strength * non_active * (excess / (floor + 1e-8))), 0.15, 1.0)
    return x - x_low + (x_low * gain)


def transient_cluster_suppress(
    x: np.ndarray, sr: int, band: Tuple[float, float], strength: float
) -> np.ndarray:
    nyq = sr / 2.0
    b, a = sig.butter(2, [band[0] / nyq, min(band[1], nyq - 1.0) / nyq], btype="band")  # type: ignore[assignment]
    filtered = sig.filtfilt(b, a, x)
    env_s = moving_average(np.abs(np.asarray(sig.hilbert(filtered))), int(0.025 * sr))  # type: ignore[arg-type]
    limit = np.percentile(env_s, 92)
    gain = np.ones_like(env_s)
    over = env_s > limit
    gain[over] = 1.0 / (1.0 + strength * (env_s[over] / (limit + 1e-8) - 1.0))
    return x * gain


def remove_baseline_morphology(x: np.ndarray, sr: int, win_sec: float = 0.15) -> np.ndarray:
    """形态学基线消除: grey_opening + grey_closing 拟合基线并减去"""
    win_len = int(win_sec * sr)
    baseline = grey_opening(x, size=win_len)
    baseline = grey_closing(baseline, size=win_len)
    return x - baseline


def denoise_heart_sound(x: np.ndarray, sr: int) -> np.ndarray:
    """新版降噪主函数"""
    # 1. 形态学基线消除
    x_stable = remove_baseline_morphology(x, sr, win_sec=BASELINE_WIN_SEC)

    # 2. 带通 (用于回混)
    x_band = butter_bandpass_zero_phase(x_stable, sr, LOW_HZ, HIGH_HZ, order=4)

    # 3. Activity 掩码 (25Hz 高通脱钩)
    x_for_mask = butter_bandpass_zero_phase(x_stable, sr, MASK_LOW_HZ, HIGH_HZ, order=2)
    activity = compute_activity_mask(x_for_mask, sr)

    # 4. 小波降噪 (在基线消除后的全频信号上)
    y = wavelet_denoise_adaptive(
        x_stable, activity,
        level=WAVELET_LEVEL,
        base_scale=BASE_THRESHOLD_SCALE,
        detail_weights=DETAIL_THRESHOLD_WEIGHTS
    )

    # 5. 后处理
    if USE_NOISE_GATE:
        y = noise_gate_inactive_only(y, sr, activity, NOISE_GATE_STRENGTH, NOISE_GATE_FLOOR_RATIO)
    if USE_LOWFREQ_FLOOR_SUPPRESS:
        y = suppress_lowfreq_floor(y, sr, activity, LOWFREQ_FLOOR_BAND, LOWFREQ_FLOOR_STRENGTH)
    if USE_CLUSTER_SUPPRESS:
        y = transient_cluster_suppress(y, sr, band=(25.0, HIGH_HZ), strength=CLUSTER_SUPPRESS_STRENGTH)

    # 6. Active 区原信号回混
    alpha = ACTIVE_BLEND_MIN + (ACTIVE_BLEND_MAX - ACTIVE_BLEND_MIN) * np.clip(activity, 0.0, 1.0)
    y = (1.0 - alpha) * y + alpha * x_band

    # 7. 最终带通收尾
    return butter_bandpass_zero_phase(y, sr, LOW_HZ, HIGH_HZ, order=2)


# =========================
# 4) 画图辅助模块 (对齐旧版)
# =========================
def limit_signal_for_plot(x: np.ndarray, sr: int, max_seconds: float):
    max_n = int(max_seconds * sr)
    return x[:max_n]


def spectrum_db(x: np.ndarray, sr: int):
    if len(x) < 8:
        return np.array([0.0]), np.array([-120.0])
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    spec_db = 20 * np.log10(np.abs(np.fft.rfft(x)) + 1e-12)
    return freqs, spec_db


def reconstruct_each_component(coeffs, wavelet_name: str):
    n_parts = len(coeffs)
    approx_signal = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet_name)
    detail_signals = []
    level = n_parts - 1
    for i in range(1, n_parts):
        tmp = [np.zeros_like(coeffs[0])]
        for j in range(1, n_parts):
            tmp.append(coeffs[j] if j == i else np.zeros_like(coeffs[j]))
        detail_signals.append((f"D{level - i + 1}", pywt.waverec(tmp, wavelet_name)))
    return approx_signal, detail_signals


def plot_basic_compare(x_b: np.ndarray, x_a: np.ndarray, sr: int, save_path: str):
    xb = limit_signal_for_plot(x_b, sr, MAX_SECONDS_TO_SHOW)
    xa = limit_signal_for_plot(x_a, sr, MAX_SECONDS_TO_SHOW)
    tb = np.arange(len(xb)) / sr
    ta = np.arange(len(xa)) / sr

    fb, sb = spectrum_db(xb, sr)
    fa, sa = spectrum_db(xa, sr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("滤波前后对比 (V2 升级版)", fontsize=16)

    axes[0, 0].plot(tb, xb)
    axes[0, 0].set_title("滤波前 - 时域")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(ta, xa)
    axes[0, 1].set_title("滤波后 - 时域")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(fb, sb)
    axes[1, 0].set_title("滤波前 - 频谱")
    axes[1, 0].set_xlim(0, 300)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(fa, sa)
    axes[1, 1].set_title("滤波后 - 频谱")
    axes[1, 1].set_xlim(0, 300)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_wavelet_compare(x_b: np.ndarray, x_a: np.ndarray, sr: int, wavelet_name: str, level: int, save_path: str):
    wavelet = pywt.Wavelet(wavelet_name)  # type: ignore[attr-defined]
    use_level = min(pywt.dwt_max_level(len(x_b), wavelet.dec_len), level)

    coeffs_b = pywt.wavedec(x_b, wavelet_name, level=use_level)
    coeffs_a = pywt.wavedec(x_a, wavelet_name, level=use_level)

    approx_b, details_b = reconstruct_each_component(coeffs_b, wavelet_name)
    approx_a, details_a = reconstruct_each_component(coeffs_a, wavelet_name)

    comps_b = [(f"A{use_level}", approx_b)] + details_b
    comps_a = [(f"A{use_level}", approx_a)] + details_a

    n_rows = len(comps_b)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.2 * n_rows), sharex=False)
    fig.suptitle(f"小波分量前后对比 ({wavelet_name}, 分解层数={use_level})", fontsize=16)

    for i, ((name_b, sig_b), (name_a, sig_a)) in enumerate(zip(comps_b, comps_a)):
        sig_b_show = limit_signal_for_plot(sig_b, sr, MAX_SECONDS_TO_SHOW)
        sig_a_show = limit_signal_for_plot(sig_a, sr, MAX_SECONDS_TO_SHOW)
        tb = np.arange(len(sig_b_show)) / sr
        ta = np.arange(len(sig_a_show)) / sr

        axes[i, 0].plot(tb, sig_b_show)
        axes[i, 0].set_title(f"滤波前 - {name_b}")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(ta, sig_a_show)
        axes[i, 1].set_title(f"滤波后 - {name_a}")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =========================
# 5) 主执行流程
# =========================
def main():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("心音小波降噪 V2 (升级版)")
    print("=" * 60)
    print(f"输入:   {INPUT_WAV}")
    print(f"输出:   {OUTPUT_WAV}")
    print(f"图表:   {OUTPUT_PLOT_DIR}")
    print(f"重采样: {'%d Hz' % TARGET_SR if TARGET_SR else '保持原始采样率'}")
    print(f"带通:   {LOW_HZ}-{HIGH_HZ} Hz")
    print(f"基线消除: 形态学 grey_open+close ({BASELINE_WIN_SEC}s窗)")
    print(f"Activity: {MASK_LOW_HZ}Hz高通脱钩")
    print()

    print("1. 正在加载音频...")
    sr_before, x_before = load_and_resample(INPUT_WAV, TARGET_SR)
    print(f"   采样率={sr_before}, 长度={len(x_before)/sr_before:.1f}s")

    print("2. 正在进行小波降噪...")
    x_after = denoise_heart_sound(x_before, sr_before)
    print("   降噪完成")

    print("3. 正在保存降噪音频...")
    save_audio(OUTPUT_WAV, sr_before, x_after)

    print("4. 正在生成对比图表...")
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    plot_1_path = os.path.join(OUTPUT_PLOT_DIR, "1_基本时频域对比.png")
    plot_2_path = os.path.join(OUTPUT_PLOT_DIR, "2_小波分解层对比.png")
    plot_basic_compare(x_before, x_after, sr_before, plot_1_path)
    plot_wavelet_compare(x_before, x_after, sr_before, PLOT_WAVELET_NAME, WAVELET_LEVEL, plot_2_path)

    # ── 数值评估 ──
    from scipy.signal import butter, filtfilt
    def band_rms(sig, lo, hi):
        nyq = sr_before / 2
        b, a = butter(2, [lo / nyq, hi / nyq], btype="band")  # type: ignore[assignment]
        return float(np.sqrt(np.mean(filtfilt(b, a, sig) ** 2)))

    env = np.abs(x_after)
    silent_mask = env < np.percentile(env, 20)
    noise_after = float(np.sqrt(np.mean(x_after[silent_mask] ** 2))) if silent_mask.sum() > 0 else 0
    noise_before = float(np.sqrt(np.mean(x_before[silent_mask] ** 2))) if silent_mask.sum() > 0 else 0

    print(f"\n  数值评估:")
    print(f"    静音区RMS:    {noise_before:.6f} -> {noise_after:.6f}  ({'%.1f'%(100*(1-noise_after/max(noise_before,1e-8)))}% 抑制)")
    print(f"    20-200Hz能量: {band_rms(x_before, 20, 200):.6f} -> {band_rms(x_after, 20, 200):.6f}")
    print(f"    10-30Hz残余:  {band_rms(x_after, 10, 30):.6f}")

    print(f"\n[完成] 降噪音频: {OUTPUT_WAV}")
    print(f"[完成] 对比图1:  {plot_1_path}")
    print(f"[完成] 对比图2:  {plot_2_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
