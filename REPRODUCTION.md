# 训练复现说明（db6 论文降噪法）

实验已整体翻新，统一采用论文 II-B 的 db6 降噪方法；此前旧实验（Haar）的数据与图表不再作为
依据。请按下面步骤从头复现「降噪 → 特征提取 → 训练评估」完整流程，即可得到规范化的结果。

## 一、技术栈

见 `requirements.txt`，环境依赖：

`numpy` `scipy` `pandas` `PyWavelets` `opensmile` `librosa` `soundfile`
`scikit-learn` `matplotlib` `joblib`

```bash
pip install -r requirements.txt
```

## 二、数据准备

Yaseen 公开数据集 1,000 条（AS / MR / MS / MVP / N 各 200，采样率 8,000 Hz）。
放到 `Feature Extraction/001_paper015/{cat}/`，并把脚本顶部的 `BASE_DIR` / `PROJECT_DIR`
改为你的实际路径。

## 三、三步复现

```bash
python denoising/batch_denoise_paper_dataset.py              # 1) db6 降噪 → {cat}_denoised/
python feature_extraction/extract_features_paper_denoised.py # 2) 提特征 → 5 个 CSV
python training/train_paper_denoised.py                      # 3) 训练 + 评估 → out_paper015_dbp/
```

## 四、产物

训练脚本输出到 `out_paper015_dbp/`：

- `result_summary.json` —— 全部指标数字汇总
- `exp_data.joblib` —— 完整训练过程数据（X/y、切分、CV 曲线、support/ranking、prediction）
- `model_deploy.joblib` —— 部署用模型 dict
- `fig_rfecv_curve.png` / `fig_confusion.png` / `fig_roc.png`

## 五、方法说明

- **降噪（论文 II-B）**：DC 去除 → 幅值削波(99.5 分位) → 4 阶 Butterworth 20–500 Hz 零相位
  → 5 层 db6 小波 + BayesShrink 软阈值 → Hilbert 包络增强(40 ms，增益 2.0) → 65%/35% 混音
  → RMS 归一化。参数依据见 `denoising/README.md`。
- **特征**：openSMILE `ComParE_2016 / Functionals`，共 6,373 维。
- **训练协议**：80/20 分层切分（`stratify`，`seed=42`）；`LinearSVC(C=1.0, max_iter=2000)`
  + `StandardScaler`；`RFECV(step=100, cv=StratifiedKFold(5, shuffle, seed=42), scoring="f1_macro")`。
- **参数依据**：论文约定 + 经验默认（`C=1.0` 为 sklearn 默认；`max_iter=2000` 确保收敛；
  `step=100` 为 6,373 维搜索的粗网格折中；5 折分层 CV + macro-F1 应对类别均衡）。