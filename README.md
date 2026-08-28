# 基于机器学习的移动听诊系统用于心脏瓣膜病筛查 — 代码仓库

本仓库提供论文《A Lightweight Intelligent Auscultation System for Valvular Heart Disease Screening》的**完整可复现代码**（不含数据）。系统以单通道电子听诊器采集的心音为输入，完成「预处理 → OpenSMILE 特征提取 → RFECV 降维 → 线性 SVM 五分类 → SHAP 可解释性分析」全流程，实现 5 类心脏瓣膜病（N / AS / MR / MS / MVP）的自动识别。

## 一、结果摘要（Exp1，纯 paper 数据集，1,000 样本）

| 模型 | 特征维度 | 准确率 | Macro-F1 |
|---|---|---|---|
| Baseline（线性 SVM） | 6,373 | 96.50% | 0.9648 |
| RFECV-SVM | 173（97.3% 降维） | 97.50% | 0.9750 |

RFECV 5 折交叉验证最佳 macro-F1 = 0.9826。

## 二、数据处理流程

```
① 降噪            ② 特征提取              ③ 训练 + 降维           ④ 结果图表 + 指标
batch_denoise     extract_features        run_exp1.py            gen_feature_reduction.py
_paper.py    →    _paper_denoised.py  →   (Baseline + RFECV)  →  (table_metrics.txt + 图)
（V2 算法）       （OpenSMILE 6373 维）     （保存 exp1_data_full.joblib）
```

## 三、目录结构

```
paper_code/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── denoising/                         # 降噪模块（详见 denoising/README.md）
│   ├── denoising_v2.py                # V2 降噪算法（数据集实际采用）
│   ├── batch_denoise_paper.py         # V2 批量降噪，输出 *_denoised/*.wav
│   ├── denoising.py                   # V1 降噪算法（仅用于 042 展示图）
│   └── plot_042.py                    # 生成降噪前后对比展示图（042 示例）
├── feature_extraction/
│   └── extract_features_paper_denoised.py   # OpenSMILE ComParE_2016 特征提取
└── training/
    ├── run_exp1.py                    # 纯 paper 训练：Baseline + RFECV
    └── gen_feature_reduction.py       # 生成最终图表 + table_metrics.txt
```

## 四、复现步骤

### 0. 准备数据

- **示例数据（已随仓库提供）**：`denoising/example_data/synth_000042_mix.wav`，可直接运行 `denoising/plot_042.py` 复现降噪展示图，无需下载。
- **完整数据集（不随仓库提供）**：Yaseen 等人公开数据集（`1,000 recordings, 5 classes`，4 kHz 单声道 WAV）。下载后按 `AS / MR / MS / MVP / N` 五个子目录放置，每类 200 个 wav。
- 目录约定：将数据集放在 `Feature Extraction/001_paper015/` 下（或修改各脚本顶部的 `PROJECT_DIR` / `BASE_DIR` 路径指向你的目录）。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 批量降噪（V2）

```bash
python denoising/batch_denoise_paper.py
```

对 `001_paper015/{AS,MR,MS,MVP,N}` 下所有 wav 降噪，输出到对应 `*_denoised/` 目录。

### 3. 特征提取（OpenSMILE）

```bash
python feature_extraction/extract_features_paper_denoised.py
```

输出 `extracted_features_{AS,MR,MS,MVP,N}_paper_denoised.csv`（每样本 6,373 维）。

### 4. 训练 + RFECV 降维（纯 paper）

```bash
python training/run_exp1.py
```

输出到 `out_paper015_rfecv/`：`exp1_data_full.joblib`（完整训练/测试数据与模型）、`rfecv_result_exp1.joblib`，以及 RFECV 曲线、混淆矩阵、ROC、特征饼图。

### 5. 生成最终图表 + 性能指标

```bash
python training/gen_feature_reduction.py
```

读取 `exp1_data_full.joblib`，输出到 `out_feature_reduction/`：`table_metrics.txt`（论文结果表）以及 RFECV 曲线、混淆矩阵、ROC、特征饼图、SHAP bar / beeswarm / 瀑布图。

## 五、重要说明

1. **降噪展示图 vs 数据集降噪（算法不同，分开说明）**：
   - 论文中的降噪效果展示图（`042_timefreq.png` / `042_wavelet.png`）由 `denoising/plot_042.py` 生成，使用的是 **V1 降噪算法**（`denoising.py`，28–240 Hz 带通 + 5 层 Haar 小波），示例输入为合成数据集 042 号（已随仓库提供，见 `denoising/example_data/`）。
   - 论文实际训练数据集采用 **V2 降噪算法**（`denoising_v2.py` / `batch_denoise_paper.py`，25–500 Hz 带通 + 形态学基线消除 + 5 层 Haar 小波自适应阈值）。
   - 两者属于不同版本的降噪实现，请勿混淆。

2. **硬编码路径**：`batch_denoise_paper.py`、`extract_features_paper_denoised.py`、`run_exp1.py` 中的 `PROJECT_DIR` / `BASE_DIR` 仍为 Windows 绝对路径，移植时需改为你的实际路径；`plot_042.py` 已改为基于脚本目录的相对路径，无需修改。

3. **数据不随仓库提供**：本仓库除一条示例音频（`denoising/example_data/synth_000042_mix.wav`）外，不含任何音频、特征 CSV、模型缓存（`.joblib`）文件，见 `.gitignore`。
