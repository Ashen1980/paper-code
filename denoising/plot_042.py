"""单独为 synth_000042_mix.wav 生成两张对比图 — 大字体。"""
import os, numpy as np, pywt
import scipy.signal as spsig
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\人工混合数据集\synth_mix\synth_000042_mix.wav"
OUT = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\人工混合数据集\synth_mix_plots_042"
os.makedirs(OUT, exist_ok=True)

LOW_HZ, HIGH_HZ = 28.0, 240.0
WAVELET_LEVEL, PLOT_WAVELET_NAME, MAX_SEC, FIG_DPI = 5, "db6", 8.0, 150
BASE_THRESHOLD_SCALE, DETAIL_THRESHOLD_WEIGHTS = 0.38, [1.00, 0.85, 0.60, 0.38, 0.24]
ACTIVE_THR_MIN, ACTIVE_THR_MAX = 0.65, 1.45
ACTIVE_BLEND_MIN, ACTIVE_BLEND_MAX = 0.02, 0.12
USE_NOISE_GATE, NOISE_GATE_STRENGTH, NOISE_GATE_FLOOR_RATIO = True, 0.24, 1.55
USE_CLUSTER_SUPPRESS, CLUSTER_SUPPRESS_STRENGTH = True, 0.16
USE_LOWFREQ_FLOOR_SUPPRESS, LOWFREQ_FLOOR_STRENGTH, LOWFREQ_FLOOR_BAND = True, 0.35, (20.0, 80.0)
TITLE_SZ = 36
FIG1_LABEL_SZ, FIG1_TICK_SZ = 46, 40
FIG2_TITLE_SZ, FIG2_LABEL_SZ, FIG2_TICK_SZ = 56, 56, 50

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

# ── V1 denoise ──
def load_audio(p):
    sr, x = wavfile.read(p)
    if x.dtype == np.int16: y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32: y = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating): y = x.astype(np.float64)
    else: raise TypeError
    if y.ndim == 2: y = np.mean(y, axis=1)
    return sr, y

def butter_bandpass_zero_phase(x, sr, low, high, order=4):
    nyq = sr / 2.0
    b, a = spsig.butter(order, [max(0.5, low) / nyq, min(high, nyq - 1.0) / nyq], btype="band")
    return spsig.filtfilt(b, a, x).astype(np.float64)

def moving_average(x, win):
    return np.convolve(x, np.ones(max(1, int(win))) / max(1, int(win)), mode="same")

def compute_activity_mask(x_band, sr):
    env = moving_average(np.abs(spsig.hilbert(x_band)), int(0.03 * sr))
    p10, p95 = np.percentile(env, 10), np.percentile(env, 95)
    act = np.clip((env - p10) / max(p95 - p10, 1e-8), 0.0, 1.0)
    return np.clip(moving_average(act, int(0.05 * sr)), 0.0, 1.0)

def _pad_even(x):
    return (x, 0) if len(x) % 2 == 0 else (np.pad(x, (0, 1), mode="edge"), 1)

def haar_wavedec(x, level=5):
    a, details, pads = x.astype(np.float64).copy(), [], []
    inv = 1.0 / np.sqrt(2.0)
    for _ in range(level):
        if len(a) < 4: break
        ap, pd = _pad_even(a)
        details.append((ap[0::2] - ap[1::2]) * inv)
        pads.append(pd)
        a = (ap[0::2] + ap[1::2]) * inv
    return a, details, pads

def haar_waverec(approx, details, pads):
    a, inv = approx.astype(np.float64).copy(), 1.0 / np.sqrt(2.0)
    for d, pd in zip(details[::-1], pads[::-1]):
        out = np.empty(d.size * 2, dtype=np.float64)
        out[0::2] = (a + d) * inv
        out[1::2] = (a - d) * inv
        a = out[:out.size - pd] if pd else out
    return a

def downsample_profile(profile, target_len):
    idx = np.linspace(0, len(profile), target_len + 1, dtype=int)
    return np.array([np.mean(profile[idx[i]:idx[i + 1]]) for i in range(target_len)])

def wavelet_denoise_adaptive(x, sr, activity, level=5, base_scale=0.38, detail_weights=None):
    if detail_weights is None: detail_weights = [1.00, 0.85, 0.60, 0.38, 0.24]
    approx, details, pads = haar_wavedec(x, level=level)
    sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
    proc = []
    for i, detail in enumerate(details):
        w = detail_weights[min(i, len(detail_weights) - 1)]
        thr = base_scale * sigma * np.sqrt(2.0 * np.log(len(detail) + 1.0)) * w
        thr *= (ACTIVE_THR_MAX - (ACTIVE_THR_MAX - ACTIVE_THR_MIN) * downsample_profile(activity, len(detail)))
        proc.append(np.sign(detail) * np.maximum(np.abs(detail) - thr, 0.0))
    return haar_waverec(approx, proc, pads)[:len(x)]

def noise_gate_inactive_only(x, sr, activity, strength, floor_ratio):
    env_s = moving_average(np.abs(spsig.hilbert(x)), int(0.02 * sr))
    excess = np.maximum(np.percentile(env_s, 20) * floor_ratio - env_s, 0.0) / (np.percentile(env_s, 20) * floor_ratio + 1e-12)
    return x * np.clip(1.0 - strength * (1.0 - np.clip(activity, 0.0, 1.0)) * excess, 0.78, 1.0)

def suppress_lowfreq_floor(x, sr, activity, band, strength):
    nyq = sr / 2.0
    b, a = spsig.butter(2, [max(1.0, band[0]) / nyq, min(band[1], nyq - 1.0) / nyq], btype="band")
    x_low = spsig.filtfilt(b, a, x)
    env_s = moving_average(np.abs(spsig.hilbert(x_low)), int(0.06 * sr))
    non = 1.0 - np.clip(activity, 0.0, 1.0)
    floor = np.percentile(env_s, 35)
    excess = np.maximum(env_s - floor, 0.0)
    gain = np.clip(1.0 / (1.0 + strength * non * (excess / (floor + 1e-8))), 0.72, 1.0)
    return x - x_low + (x_low * gain)

def transient_cluster_suppress(x, sr, band, strength):
    nyq = sr / 2.0
    b, a = spsig.butter(2, [band[0] / nyq, min(band[1], nyq - 1.0) / nyq], btype="band")
    env_s = moving_average(np.abs(spsig.hilbert(spsig.filtfilt(b, a, x))), int(0.025 * sr))
    limit = np.percentile(env_s, 92)
    over = env_s > limit
    gain = np.ones_like(env_s)
    gain[over] = 1.0 / (1.0 + strength * (env_s[over] / (limit + 1e-8) - 1.0))
    return x * gain

def denoise_heart_sound(x, sr):
    x_band = butter_bandpass_zero_phase(x, sr, LOW_HZ, HIGH_HZ, order=4)
    activity = compute_activity_mask(x_band, sr)
    y = wavelet_denoise_adaptive(x_band, sr, activity, level=WAVELET_LEVEL, base_scale=BASE_THRESHOLD_SCALE, detail_weights=DETAIL_THRESHOLD_WEIGHTS)
    if USE_NOISE_GATE: y = noise_gate_inactive_only(y, sr, activity, NOISE_GATE_STRENGTH, NOISE_GATE_FLOOR_RATIO)
    if USE_LOWFREQ_FLOOR_SUPPRESS: y = suppress_lowfreq_floor(y, sr, activity, LOWFREQ_FLOOR_BAND, LOWFREQ_FLOOR_STRENGTH)
    if USE_CLUSTER_SUPPRESS: y = transient_cluster_suppress(y, sr, (25.0, HIGH_HZ), CLUSTER_SUPPRESS_STRENGTH)
    alpha = ACTIVE_BLEND_MIN + (ACTIVE_BLEND_MAX - ACTIVE_BLEND_MIN) * np.clip(activity, 0.0, 1.0)
    return butter_bandpass_zero_phase((1.0 - alpha) * y + alpha * x_band, sr, LOW_HZ, HIGH_HZ, order=2)

# ── Plot ──
def limit_signal(x, sr, max_sec):
    return x[:int(max_sec * sr)]

def spectrum_db(x, sr):
    if len(x) < 8: return np.array([0.0]), np.array([-120.0])
    return np.fft.rfftfreq(len(x), d=1.0 / sr), 20 * np.log10(np.abs(np.fft.rfft(x)) + 1e-12)

def set_fonts(ax, label_sz, tick_sz, title_sz=None):
    ax.title.set_fontsize(title_sz if title_sz is not None else TITLE_SZ)
    ax.xaxis.label.set_fontsize(label_sz)
    ax.yaxis.label.set_fontsize(label_sz)
    ax.tick_params(axis="both", which="major", labelsize=tick_sz)

def reconstruct_each_component(coeffs, wname):
    approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wname)
    details = []
    lv = len(coeffs) - 1
    for i in range(1, len(coeffs)):
        tmp = [np.zeros_like(coeffs[0])] + [coeffs[j] if j == i else np.zeros_like(coeffs[j]) for j in range(1, len(coeffs))]
        details.append((f"D{lv - i + 1}", pywt.waverec(tmp, wname)))
    return approx, details

sr, x_raw = load_audio(SRC)
x_den = denoise_heart_sound(x_raw, sr)
print(f"042 loaded: sr={sr}, len={len(x_raw)/sr:.1f}s, denoised")

# Fig1
xb, xa = limit_signal(x_raw, sr, MAX_SEC), limit_signal(x_den, sr, MAX_SEC)
tb, ta = np.arange(len(xb)) / sr, np.arange(len(xa)) / sr
fb, sb = spectrum_db(xb, sr); fa, sa = spectrum_db(xa, sr)

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.subplots_adjust(left=0.12, right=0.95, hspace=0.45, wspace=0.30)
for ax, t, y, xl in [(axes[0,0], tb, xb, None), (axes[0,1], ta, xa, None),
                       (axes[1,0], fb, sb, 1000), (axes[1,1], fa, sa, 1000)]:
    ax.plot(t, y, lw=1.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    if xl: ax.set_xlim(0, xl); ax.set_xticks(np.arange(0, xl + 1, 200))
axes[0,0].set_title("Before - Time Domain"); axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("Amplitude")
axes[0,1].set_title("After - Time Domain"); axes[0,1].set_xlabel("Time (s)"); axes[0,1].set_ylabel("Amplitude")
axes[1,0].set_title("Before - Spectrum"); axes[1,0].set_xlabel("Frequency (Hz)"); axes[1,0].set_ylabel("Magnitude (dB)")
axes[1,1].set_title("After - Spectrum"); axes[1,1].set_xlabel("Frequency (Hz)"); axes[1,1].set_ylabel("Magnitude (dB)")
for ax in axes.flat: set_fonts(ax, FIG1_LABEL_SZ, FIG1_TICK_SZ)
# Override yticks after set_fonts
axes[0,0].set_yticks([-1, 0, 1])
axes[0,1].set_yticks([-1, 0, 1])
axes[1,0].set_yticks([-60, 0, 60])
axes[1,0].set_ylim(-60, 60)
axes[1,1].set_yticks([-60, 0, 60])
axes[1,1].set_ylim(-60, 60)
p1 = os.path.join(OUT, "042_timefreq.png")
fig.savefig(p1, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  {p1}  ({os.path.getsize(p1)//1024} KB)")

# Fig2
wv = pywt.Wavelet(PLOT_WAVELET_NAME)
ulv = min(pywt.dwt_max_level(len(x_raw), wv.dec_len), WAVELET_LEVEL)
cb, ca = pywt.wavedec(x_raw, PLOT_WAVELET_NAME, level=ulv), pywt.wavedec(x_den, PLOT_WAVELET_NAME, level=ulv)
ab, db = reconstruct_each_component(cb, PLOT_WAVELET_NAME)
aa, da = reconstruct_each_component(ca, PLOT_WAVELET_NAME)
cb_comp = [(f"A{ulv}", ab)] + db; ca_comp = [(f"A{ulv}", aa)] + da
nr = len(cb_comp)
fig2, axes2 = plt.subplots(nr, 2, figsize=(32, 5.5 * nr))
fig2.subplots_adjust(left=0.15, right=0.92, bottom=0.08, top=0.95, hspace=0.5)
for j, ((nb, sb), (na, sa)) in enumerate(zip(cb_comp, ca_comp)):
    for ax, nm, sig in [(axes2[j, 0], nb, limit_signal(sb, sr, MAX_SEC)),
                         (axes2[j, 1], na, limit_signal(sa, sr, MAX_SEC))]:
        ax.plot(np.arange(len(sig)) / sr, sig, lw=1.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title(f"{'Before' if nm == nb else 'After'} - {nm}")
        if j < 5:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        set_fonts(ax, FIG2_LABEL_SZ, FIG2_TICK_SZ, FIG2_TITLE_SZ)
fig2.subplots_adjust(wspace=0.35)
p2 = os.path.join(OUT, "042_wavelet.png")
fig2.savefig(p2, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  {p2}  ({os.path.getsize(p2)//1024} KB)")
print("Done.")
