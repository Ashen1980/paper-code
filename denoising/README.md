# 降噪模块说明

本模块包含心音信号降噪算法与效果展示。随仓库提供**一条示例合成数据**（`example_data/synth_000042_mix.wav`，约 62 KB）用于降噪演示，其余数据（paper 数据集、合成数据集）不随仓库提供。

## 论文降噪方法（db6，当前数据集实际采用）

`batch_denoise_paper_dataset.py` 复现论文 II-B 的降噪方法，处理顺序如下：

1. 单声道（paper 数据原始即为 8,000 Hz，无需重采样）
2. DC 去除（减均值）
3. 幅值削波（99.5 分位对称削波，抑制尖峰）
4. 4 阶 Butterworth 带通 20–500 Hz（零相位 `filtfilt`）
5. 5 层 db6 小波 + BayesShrink 自适应软阈值
6. Hilbert 包络增强（40 ms 滑动平均，增益上限 2.0）
7. 65%/35% 混音（65% 去噪 + 35% 原带通信号）
8. RMS 归一化

关键参数及选择依据：

| 参数 | 取值 | 依据 |
|------|------|------|
| 采样率 | 8000 Hz | paper 数据集原始采样率（正文 E 节 4000 Hz 为笔误） |
| 小波基 | db6 | 紧支、正则性好，6 阶消失矩，适合心音非平稳瞬态 |
| 分解层数 | 5 | 8000 Hz 下最低逼近子带覆盖 ~125 Hz（对应 A5） |
| 阈值 | BayesShrink β=1.0 | 公式(1)(2)：σ=MAD(d1)/0.6745，逐层自适应软阈值 |
| 带通 | 20–500 Hz | 覆盖心音与病理杂音频带，去呼吸/高频噪声 |
| 混音 | 65% 去噪 + 35% 原带通 | 论文固定比例，保留细节防过度去噪 |

处理对象：`001_paper015/{AS,MR,MS,MVP,N}/*.wav`，输出到对应 `*_denoised/`。

## 文件清单

| 文件 | 算法版本 | 作用 |
|---|---|---|
| `batch_denoise_paper_dataset.py` | db6（论文 II-B） | **批量降噪 paper 数据集（当前采用）** |
| `denoising_v2.py` / `batch_denoise_paper.py` | V2（Haar） | 旧算法（25–500 Hz + 形态学基线消除 + Haar 自适应阈值），已弃用 |
| `denoising.py` / `plot_042.py` | V1（Haar） | 旧算法（28–240 Hz），仅用于生成 042 展示图 |

## 运行方式

```bash
# 批量降噪（db6，数据集）
python denoising/batch_denoise_paper_dataset.py

# 生成 042 展示图（V1）
python denoising/plot_042.py
```

## 说明

- 数据集降噪已从旧 V2（Haar）切换到 db6 方法，重训结果见仓库根目录 README。
- `batch_denoise_paper_dataset.py` 为自包含脚本（内嵌 db6 算法全部函数），可独立运行；其 `BASE` 为绝对路径，需按你的目录修改。
- `plot_042.py` 已改为基于脚本所在目录的相对路径（示例数据与输出均在 `example_data/` 下，无需改路径）。