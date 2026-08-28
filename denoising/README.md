# 降噪模块说明

本模块包含心音信号降噪算法与效果展示。随仓库提供**一条示例合成数据**（`example_data/synth_000042_mix.wav`，约 62 KB）用于降噪演示，其余数据（paper 数据集、合成数据集）不随仓库提供。

## 文件清单

| 文件 | 算法版本 | 作用 |
|---|---|---|
| `denoising_v2.py` | V2 | 降噪算法本体（论文数据集实际采用） |
| `batch_denoise_paper.py` | V2 | 批量降噪 paper 数据集（5 类各 200 个 wav） |
| `denoising.py` | V1 | 降噪算法本体（仅用于生成 042 展示图） |
| `plot_042.py` | V1 | 生成降噪前后对比展示图（时频域 + 小波分解） |

## 两种降噪算法（重要：需区分）

### V2（数据集降噪，论文训练所用）

`denoising_v2.py` / `batch_denoise_paper.py`，参数：

- 重采样至 8,000 Hz
- 形态学基线消除（grey opening/closing，窗长 0.15 s）
- 4 阶 Butterworth 带通 25–500 Hz（零相位 `filtfilt`）
- 5 层 Haar 小波自适应阈值（`BASE_THRESHOLD_SCALE=0.38`，`DETAIL_THRESHOLD_WEIGHTS=[1.00, 0.85, 0.65, 0.45, 0.30]`）
- 噪声门（0.24 / 1.55）、瞬态簇抑制（0.16）、低频地板抑制（2.5，10–60 Hz）
- RMS 归一化

处理对象：`001_paper015/{AS,MR,MS,MVP,N}/*.wav`，输出到对应 `*_denoised/`。

### V1（展示图降噪，仅用于 042 示例）

`denoising.py` / `plot_042.py`，参数：

- 4 阶 Butterworth 带通 28–240 Hz（零相位）
- 5 层 Haar 小波自适应阈值（`BASE_THRESHOLD_SCALE=0.38`，`DETAIL_THRESHOLD_WEIGHTS=[1.00, 0.85, 0.60, 0.38, 0.24]`）
- 噪声门（0.24 / 1.55）、瞬态簇抑制（0.16）、低频地板抑制（0.35，20–80 Hz）

处理对象：示例数据 `example_data/synth_000042_mix.wav`（已随仓库提供），输出两张对比图到 `example_data/output/`。

## 运行方式

```bash
# 批量降噪（V2，数据集）
python denoising/batch_denoise_paper.py

# 生成 042 展示图（V1）
python denoising/plot_042.py
```

## 说明

- 小波分解展示图中使用的 `db6`（`PLOT_WAVELET_NAME`）仅用于**可视化分解层**，实际降噪阈值处理使用 **Haar** 小波。
- `batch_denoise_paper.py` 为自包含脚本（已内嵌 V2 算法全部函数），可独立运行，无需 `import denoising_v2`。
- `plot_042.py` 已改为基于脚本所在目录的相对路径（示例数据与输出均在 `example_data/` 下，无需改路径）；`batch_denoise_paper.py` 的 `BASE_DIR` 仍为绝对路径，需按你的目录修改。
