# 训练复现与结果说明（db6 论文降噪法）

本文档回答合作者关于**模型训练**的复现、完整数据与证据问题（对应 AC08–AC10）。
AC01–AC07 为后端/部署相关问题，不在本文档范围。

---

## 一、技术栈（复现环境）

| 依赖 | 版本（最低） | 用途 |
|---|---|---|
| Python | 3.8+ | 运行环境 |
| numpy | >=1.24 | 数值计算 |
| scipy | >=1.10 | 滤波 / 重采样 / Hilbert |
| pandas | >=2.0 | 特征表读写 |
| PyWavelets | >=1.4 | db6 小波分解 + BayesShrink 阈值 |
| opensmile | >=2.5.0 | ComParE_2016 特征提取 |
| librosa | >=0.10.0 | 音频读取 |
| soundfile | >=0.12.0 | 音频写入 |
| scikit-learn | >=1.3.0 | LinearSVC / RFECV / StratifiedKFold |
| matplotlib | >=3.7.0 | 绘图 |
| joblib | >=1.3.0 | 模型与产物序列化 |

安装：`pip install -r requirements.txt`

---

## 二、三步复现

```bash
pip install -r requirements.txt
python denoising/batch_denoise_paper_dataset.py              # 1) db6 降噪
python feature_extraction/extract_features_paper_denoised.py # 2) 提特征
python training/train_paper_denoised.py                      # 3) 训练 + 评估
```

数据准备：Yaseen 公开数据集 1,000 条（AS / MR / MS / MVP / N 各 200，采样率 8,000 Hz）。
放到 `Feature Extraction/001_paper015/{cat}/`，并把脚本顶部的 `BASE_DIR` / `PROJECT_DIR` 改为实际路径。

---

## 三、完整训练数据与结果（db6 降噪法）

协议：

- 特征集：openSMILE **ComParE_2016 / Functionals**，共 6,373 维
- 切分：80/20 分层切分（`stratify`，`random_state=42`），800 训练 / 200 测试（每类 40）
- 基线：`LinearSVC(C=1.0, dual="auto", max_iter=2000)` + `StandardScaler`
- 特征选择：`RFECV(step=100, cv=StratifiedKFold(5, shuffle, seed=42), scoring="f1_macro")`

总体指标：

| 指标 | 数值 |
|---|---|
| 样本数 | 1,000（训练 800 / 测试 200） |
| 特征总维数 | 6,373 |
| Baseline（全 6,373 维）Acc | 97.50% |
| Baseline macro-F1 | 97.51% |
| RFECV 最优特征数 | 273 |
| RFECV 5 折 CV macro-F1 | 98.76% |
| 测试集 Acc | 98.50% |
| 测试集 macro-F1 | 98.50% |
| 测试集 macro-P / macro-R | 98.60% / 98.50% |

分类别（测试集，每类 40 条）：

| 类别 | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| AS | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MR | 1.0000 | 1.0000 | 1.0000 | 0.9997 |
| MS | 1.0000 | 0.9250 | 0.9610 | 0.9742 |
| MVP | 0.9302 | 1.0000 | 0.9639 | 0.9925 |
| N | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

错分情况：仅 3 条 MS 被误判为 MVP（二者同属二尖瓣病变，收缩期杂音频带接近），其余类全部正确。

完整训练过程产物（本地 `out_paper015_dbp/`，不随 git 仓库提供）：

- `result_summary.json` —— 上述数字汇总
- `exp_data.joblib` —— 完整训练过程数据（X/y、切分、`baseline`、`rfecv.support/ranking`、
  `cv_scores/cv_std/n_list`、`y_pred`、`report`、`per_class`、`selected_features`）
- `model_deploy.joblib` —— 部署模型 dict
- `fig_rfecv_curve.png` / `fig_confusion.png` / `fig_roc.png`

---

## 四、训练相关问题答复（AC08–AC10）

### AC08 — 原始 97.5% artifacts

**有（大部分）。**

- 模型 joblib：`Feature Extraction/out_paper015_rfecv/rfecv_result_exp1.joblib`（含 pipeline、
  `n_selected=173`、`support`、`ranking`、`feature_cols`）
- 完整训练数据：`out_paper015_rfecv/exp1_data_full.joblib`（含 X/y、切分、`y_pred`、
  `cv_scores`/`cv_std`/`n_list`、bl/rfecv 评估）
- 数字汇总：`out_feature_reduction/table_metrics.txt`（173 特征、Acc=0.9750、CV-F1=0.9826）
- SHAP：`out_feature_reduction/`（`fig_shap_bar_exp1.png`、`fig_shap_beeswarm_exp1.png`、
  `fig_waterfall_AS/MR_exp1.png`）与 `out_paper015_rfecv/` 同名图
- **特征 CSV 已被覆盖**：`extracted_features_{cat}_paper_denoised.csv` 于 2026-09-02 被 db6 重跑
  覆盖（现在只有 db6 版）；旧 Haar 特征 CSV 需从原始音频 + 旧降噪脚本重生成，或从
  `exp1_data_full.joblib` 的 X 矩阵还原。原始 1000 条音频仍在 `001_paper015/{cat}/`。

### AC09 — 超参数为什么这样选

**经验默认值 / 继承自原始代码，未做系统网格搜索。**

- 线性 SVM：高维小样本（6,373 维 / 800 训练）下线性核简单、可解释、不易过拟合。
- `C=1.0`：sklearn `LinearSVC` 默认值。
- `max_iter=2000`：默认 1000 偶发不收敛，提高到 2000 确保收敛。
- `RFECV step=100`：6,373 维全遍历代价过大，粗网格折中，由 CV 自动定最优特征数。
- 5 折分层 CV：常规稳健选择，分层保证每折各类比例一致。
- `macro-F1`：多分类且类别均衡，宏平均 F1 作为特征选择与评估准则。
- `seed=42` 固定：保证可复现。

### AC10 — Figures 2 / 3 / 5 / 6 provenance

- **Fig. 2**：是。基于 `denoising/example_data/synth_000042_mix.wav` + V1 可视化脚本
  （`denoising.py` / `plot_042.py`，28–240 Hz 带通 + 5 层 Haar），生成时频域对比图。
- **Fig. 3**：是（同一示例脚本、同一条 042 音频）。为小波分解图（`plot_042.py`），
  图中 `PLOT_WAVELET_NAME` 用 db6 仅作分解层可视化，实际阈值处理为 Haar；
  Fig.3 的 A5=125 Hz 即 8000 Hz 下 5 层分解最低逼近子带的反推依据。
- **Fig. 5**：195/200 混淆矩阵来自**旧 Haar 实验 Exp1**（173 特征 RFECV，测试 200 条中
  195 正确 = 97.50%）。原图 `out_paper015_rfecv/fig_cm_exp1.png`、
  `out_feature_reduction/fig_cm_exp1.png`；prediction 在 `exp1_data_full.joblib`。
- **Fig. 6**：SHAP 图来自**旧 97.5% 模型（Exp1）**，由 `training/gen_feature_reduction.py`
  基于 `exp1_data_full.joblib` 生成，产物 `out_feature_reduction/fig_shap_bar_exp1.png`、
  `fig_shap_beeswarm_exp1.png`、`fig_waterfall_AS/MR_exp1.png`。

