# 基于机器学习的移动听诊系统用于心脏瓣膜病筛查 — 代码仓库

本仓库提供论文《A Lightweight Intelligent Auscultation System for Valvular Heart Disease Screening》的**完整可复现代码**（不含数据）。系统以单通道电子听诊器采集的心音为输入，完成「预处理 → OpenSMILE 特征提取 → RFECV 降维 → 线性 SVM 五分类 → SHAP 可解释性分析」全流程，实现 5 类心脏瓣膜病（N / AS / MR / MS / MVP）的自动识别。

## 一、结果摘要（论文数据集，1,000 样本，db6 降噪）

| 模型               | 特征维度           | 准确率    | Macro-F1 |
| ---------------- | -------------- | ------ | -------- |
| Baseline（线性 SVM） | 6,373          | 97.50% | 0.9751   |
| RFECV-SVM        | 273（95.7% 降维） | 98.50% | 0.9850   |

RFECV 5 折交叉验证最佳 macro-F1 = 0.9876。

per-class F1：AS / MR / N = 1.000，MS = 0.961，MVP = 0.964（MS 与 MVP 同属二尖瓣病变，仅 3 个 MS 被误判为 MVP）。

## 二、数据处理流程

```
① 降噪（db6）           ② 特征提取                   ③ 训练 + 降维
batch_denoise          extract_features             train_paper_denoised.py
_paper_dataset.py  →   _paper_denoised.py       →   (Baseline + RFECV)
（论文 II-B 方法）        （OpenSMILE 6373 维）          （exp_data / result_summary / 部署模型）
```

## 三、目录结构

```
paper_code/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── denoising/                         # 降噪模块（详见 denoising/README.md）
│   ├── batch_denoise_paper_dataset.py # db6 降噪（论文 II-B，数据集实际采用）
│   ├── denoising_v2.py                # V2 旧算法（Haar，25–500 Hz，已弃用）
│   ├── batch_denoise_paper.py         # V2 旧批量脚本（已弃用）
│   ├── denoising.py                   # V1 旧算法（仅用于 042 展示图）
│   ├── plot_042.py                    # 降噪展示图（042 示例）
│   └── example_data/                  # 示例音频 + 输出
├── feature_extraction/
│   └── extract_features_paper_denoised.py   # OpenSMILE ComParE_2016 特征提取
└── training/
    ├── train_paper_denoised.py        # db6 数据训练：Baseline + RFECV + 部署模型
    ├── run_exp1.py                    # 训练脚本（等价协议，输出到 out_paper015_rfecv/）
    └── gen_feature_reduction.py       # 生成最终图表 + table_metrics.txt
```

## 四、复现步骤

### 0. 准备数据

* **示例数据（已随仓库提供）**：`denoising/example_data/synth_000042_mix.wav`，可直接运行 `denoising/plot_042.py` 复现降噪展示图，无需下载。

* **完整数据集（不随仓库提供）**：Yaseen 等人公开数据集（`1,000 recordings, 5 classes`，WAV）。下载后按 `AS / MR / MS / MVP / N` 五个子目录放置，每类 200 个 wav。

* 目录约定：将数据集放在 `Feature Extraction/001_paper015/` 下（或修改各脚本顶部的 `PROJECT_DIR` / `BASE_DIR` 路径指向你的目录）。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 批量降噪（db6，论文 II-B 方法）

```bash
python denoising/batch_denoise_paper_dataset.py
```

对 `001_paper015/{AS,MR,MS,MVP,N}` 下所有 wav 降噪，输出到对应 `*_denoised/` 目录。

### 3. 特征提取（OpenSMILE）

```bash
python feature_extraction/extract_features_paper_denoised.py
```

输出 `extracted_features_{AS,MR,MS,MVP,N}_paper_denoised.csv`（每样本 6,373 维）。

### 4. 训练 + RFECV 降维（db6 数据）

```bash
python training/train_paper_denoised.py
```

输出到 `out_paper015_dbp/`：`exp_data.joblib`（完整训练/测试数据与模型）、`result_summary.json`（数字汇总）、`model_deploy.joblib`（部署模型 dict），以及 RFECV 曲线、混淆矩阵、ROC。

### 5. 生成最终图表 + 性能指标（可选）

```bash
python training/gen_feature_reduction.py
```

读取训练产物，输出到 `out_feature_reduction/`：`table_metrics.txt`（论文结果表）以及 RFECV 曲线、混淆矩阵、ROC、特征饼图、SHAP bar / beeswarm / 瀑布图。

## 五、重要说明

1. **降噪算法已切换到论文 II-B 的 db6 方法**：

   * 数据集降噪现采用 `batch_denoise_paper_dataset.py`：DC 去除 → 幅值削波(99.5 分位) → 4 阶 Butterworth 带通 20–500 Hz（零相位）→ 5 层 db6 小波 BayesShrink 软阈值 → Hilbert 包络增强（40 ms，增益 2.0）→ 65%/35% 混音 → RMS 归一化。

   * 旧 V2 算法（`denoising_v2.py` / `batch_denoise_paper.py`，Haar 小波 + 25–500 Hz + 形态学基线消除）已弃用，保留仅供历史对照。

   * 旧 V1 算法（`denoising.py`，28–240 Hz + 5 层 Haar）仅用于生成 042 降噪展示图（`plot_042.py`）。

2. **硬编码路径**：`batch_denoise_paper_dataset.py`、`extract_features_paper_denoised.py`、`train_paper_denoised.py`、`run_exp1.py` 中的 `PROJECT_DIR` / `BASE_DIR` 仍为 Windows 绝对路径，移植时需改为你的实际路径；`plot_042.py` 已改为基于脚本目录的相对路径，无需修改。

3. **数据不随仓库提供**：本仓库除一条示例音频（`denoising/example_data/synth_000042_mix.wav`）外，不含任何音频、特征 CSV、模型缓存（`.joblib`）文件，见 `.gitignore`。