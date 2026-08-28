"""
从 Exp1 缓存数据（原始 1000 条数据集）生成特征降维相关全部图片。
输出到 out_feature_reduction 文件夹。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, shap, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, f1_score
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CACHE_DIR = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\out_paper015_rfecv"
OUT_DIR   = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction\out_feature_reduction"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
CLASS_NAMES = ["AS", "MR", "MS", "MVP", "N"]
COLORS_5 = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

def p(msg): print(msg, flush=True)

# ============================================================
# 加载数据
# ============================================================
p("Loading Exp1 data...")
exp = joblib.load(os.path.join(CACHE_DIR, "exp1_data_full.joblib"))

X_train = exp["X_train"]     # (800, 6373)
y_train = exp["y_train"]     # (800,)
X_test  = exp["X_test"]      # (200, 6373)
y_test  = exp["y_test"]      # (200,)
fc      = exp["fc"]          # 6373 feature names
support = exp["rfecv"]["support"]   # bool [6373], 173 True
pipeline = exp["rfecv"]["pipeline"]

cv_scores = exp["rfecv"]["cv_scores"]
cv_std    = exp["rfecv"]["cv_std"]
n_list    = exp["rfecv"]["n_list"]
optimal_score = exp["rfecv"]["optimal_score"]

X_train_sel = X_train[:, support]
X_test_sel  = X_test[:, support]
sel_names   = [fc[i] for i in range(len(fc)) if support[i]]
n_sel       = support.sum()
n_feat      = len(fc)

p(f"  Exp1: {n_feat}d → {n_sel}d,  CV best macro-F1 = {optimal_score:.4f}")

# ============================================================
# 训练 SVM（RFECV 选中特征）
# ============================================================
sc = StandardScaler().fit(X_train_sel)
X_tr_s = sc.transform(X_train_sel)
X_te_s = sc.transform(X_test_sel)

svm = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                random_state=RANDOM_SEED, max_iter=2000)
svm.fit(X_tr_s, y_train)
y_pred = svm.predict(X_te_s)
acc = (y_pred == y_test).mean()
macro_f1 = f1_score(y_test, y_pred, average="macro")

# 基线
Z_train = StandardScaler().fit_transform(X_train)
Z_test  = StandardScaler().fit(X_train).transform(X_test)
svm_bl = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                   random_state=RANDOM_SEED, max_iter=2000)
svm_bl.fit(Z_train, y_train)
y_pred_bl = svm_bl.predict(Z_test)
acc_bl = (y_pred_bl == y_test).mean()
f1_bl  = f1_score(y_test, y_pred_bl, average="macro")

# ============================================================
# 1. RFECV 优化曲线
# ============================================================
p("1/8: fig_rfecv_curve.png")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_list, cv_scores, color="#0072B2", lw=1.6)
ax.fill_between(n_list, cv_scores - cv_std, cv_scores + cv_std,
                color="#0072B2", alpha=0.12)
ax.axvline(x=n_sel, color="#D55E00", lw=1.5, ls="--",
           label=f"Optimal = {n_sel} features")
ax.scatter([n_sel], [optimal_score], color="#D55E00", s=70, zorder=5)
ax.set_xlabel("Number of features selected", fontsize=11)
ax.set_ylabel("CV macro-averaged F1", fontsize=11)
ax.set_title("RFECV Feature Selection Curve (1,000 recordings)",
             fontsize=12, fontweight="normal")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.15, lw=0.5)
ax.tick_params(labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_rfecv_curve.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 2. 混淆矩阵
# ============================================================
p("2/8: fig_confusion_matrix.png")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
for i in range(5):
    for j in range(5):
        color = "white" if cm[i, j] > cm.max() / 2 else "#222222"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(f"Confusion Matrix ({n_sel} features)\nAcc = {acc:.4f}   macro-F1 = {macro_f1:.4f}",
             fontsize=12, fontweight="normal")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_confusion_matrix.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 3. ROC 曲线
# ============================================================
p("3/8: fig_roc.png")
ds = svm.decision_function(X_te_s)
yb = label_binarize(y_test, classes=range(5))

fig, ax = plt.subplots(figsize=(5.5, 5))
for i, (lb, c) in enumerate(zip(CLASS_NAMES, COLORS_5)):
    fpr, tpr, _ = roc_curve(yb[:, i], ds[:, i])
    ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{lb} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.35)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.grid(alpha=0.15, lw=0.5)
ax.set_title("ROC Curves (1,000 recordings)", fontsize=11, fontweight="normal", pad=10)
ax.tick_params(labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_roc.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 4. 特征饼图
# ============================================================
p("4/8: fig_feature_pie.png")
PIE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
              "#E69F00", "#999999", "#AAAAAA", "#BBBBBB", "#CCCCCC"]
CAT_ORDER = ["MFCC", "audSpec_Rfilt", "audSpec", "loudness", "F0/voicing",
             "pcm_fftMag", "pcm_zcr", "pcm_RMSenergy", "jitter/shimmer", "Other"]

cats = {k: 0 for k in CAT_ORDER}
for name in sel_names:
    if "mfcc" in name: cats["MFCC"] += 1
    elif "audSpec_Rfilt" in name: cats["audSpec_Rfilt"] += 1
    elif "audSpec" in name: cats["audSpec"] += 1
    elif "loudness" in name.lower(): cats["loudness"] += 1
    elif "F0" in name or "voicing" in name or "logHNR" in name: cats["F0/voicing"] += 1
    elif "pcm_fftMag" in name: cats["pcm_fftMag"] += 1
    elif "pcm_zcr" in name: cats["pcm_zcr"] += 1
    elif "pcm_RMSenergy" in name: cats["pcm_RMSenergy"] += 1
    elif "jitter" in name or "shimmer" in name: cats["jitter/shimmer"] += 1
    else: cats["Other"] += 1

cats = {k: v for k, v in cats.items() if v > 0}
vals = list(cats.values())
colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(vals))]
total = sum(vals)
labels = []
for k, v in cats.items():
    pct = v / total * 100
    labels.append(f"{k}  {v} ({pct:.1f}%)" if pct >= 3 else f"{k} ({pct:.1f}%)")

fig, ax = plt.subplots(figsize=(10, 8))
wedges, _ = ax.pie(vals, labels=None, colors=colors, startangle=140,
                   wedgeprops={"linewidth": 0.8, "edgecolor": "white"}, radius=0.4)
for i, (wedge, label) in enumerate(zip(wedges, labels)):
    ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
    x = 0.42 * np.cos(np.deg2rad(ang))
    y = 0.42 * np.sin(np.deg2rad(ang))
    lx = 0.53 * np.cos(np.deg2rad(ang))
    ly = 0.53 * np.sin(np.deg2rad(ang))
    ha = "left" if x > 0 else "right"
    ax.annotate(label, xy=(x, y), xytext=(lx, ly),
                ha=ha, va="center", fontsize=7.5, fontweight="bold", color="#222222",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.4))
ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.62, 0.62)
ax.set_aspect("equal")
ax.set_title(f"Feature Subset Composition ({n_sel} features, 1,000 recordings)",
             fontsize=12, fontweight="normal", pad=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_feature_pie.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 5. SHAP Bar — Top 30
# ============================================================
p("5/8: fig_shap_bar.png")
bg = shap.sample(X_tr_s, 200, random_state=RANDOM_SEED)
to_exp = shap.sample(X_te_s, min(80, len(X_te_s)), random_state=RANDOM_SEED)
explainer = shap.LinearExplainer(svm, bg, feature_perturbation="interventional")
shap_vals = explainer(to_exp).values  # shape: (samples, 173, 5)
global_imp = np.mean(np.mean(np.abs(shap_vals), axis=0), axis=1)  # (173,)
top30 = np.argsort(-global_imp)[:30]

fig, ax = plt.subplots(figsize=(7.5, 7.2))
y_pos = np.arange(30)
ax.barh(y_pos, global_imp[top30][::-1], color="#0072B2", height=0.68)
ax.set_yticks(y_pos)
ax.set_yticklabels([sel_names[i] for i in top30][::-1], fontsize=6.5, fontfamily="monospace")
ax.set_xlabel("Mean |SHAP|", fontsize=11)
ax.set_title("Global SHAP Feature Importance (1,000 recordings)",
             fontsize=11, fontweight="normal", pad=10)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.15, lw=0.5)
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_shap_bar.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 6. SHAP Beeswarm — Top 15（多类别平均值）
# ============================================================
p("6/8: fig_shap_beeswarm.png")
explainer_full = shap.LinearExplainer(svm, bg, feature_perturbation="interventional")
vals_full = explainer_full(X_te_s).values  # (200, 173, 5)
vals_avg = np.mean(np.abs(vals_full), axis=2)  # (200, 173) — average across classes
top15 = np.argsort(-global_imp)[:15]
vals_top15 = vals_avg[:, top15]               # (200, 15)
top_names = [sel_names[i] for i in top15]

shap.summary_plot(vals_top15, X_te_s[:, top15],
                  feature_names=top_names, show=False, max_display=15)
fig = plt.gcf()
fig.set_size_inches(8, 5)
ax = plt.gca()
ax.set_title("SHAP Beeswarm — Top 15 Features (1,000 recordings)",
             fontsize=11, fontweight="normal", pad=10)
ax.tick_params(labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_shap_beeswarm.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# 7. 瀑布图 — AS / MR 类特异性
# ============================================================
p("7/8: fig_waterfall_AS.png, fig_waterfall_MR.png")
for cls_idx, cls_name in [(0, "AS"), (1, "MR")]:
    shap_cls = np.mean(np.abs(vals_full[:, :, cls_idx]), axis=0)  # (173,)
    top20 = np.argsort(-shap_cls)[:20]
    imp_top20 = shap_cls[top20]
    feat_names_20 = [sel_names[i] for i in top20]
    short_names = [fn if len(fn) <= 35 else fn[:32] + "..." for fn in feat_names_20]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    y_pos = np.arange(20)
    ax.barh(y_pos, imp_top20[::-1], color="#0072B2", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names[::-1], fontsize=6.5, fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP|", fontsize=11)
    ax.set_title(f"Top 20 Features for {cls_name} (1,000 recordings)",
                 fontsize=11, fontweight="normal", pad=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig_waterfall_{cls_name}.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ============================================================
# 8. 性能报告
# ============================================================
p("8/8: table_metrics.txt")
report_bl = classification_report(y_test, y_pred_bl, target_names=CLASS_NAMES, digits=4)
report_rf = classification_report(y_test, y_pred,  target_names=CLASS_NAMES, digits=4)

lines = []
lines.append("=" * 70)
lines.append("Feature Reduction Results — Exp1 (1,000 recordings)")
lines.append("=" * 70)
lines.append(f"Original features: {n_feat}")
lines.append(f"Selected features: {n_sel}  (reduction: {100*(1-n_sel/n_feat):.1f}%)")
lines.append(f"RFECV CV best macro-F1: {optimal_score:.4f}")
lines.append("")
lines.append("--- Baseline (6,373d) ---")
lines.append(f"Accuracy = {acc_bl:.4f}   Macro-F1 = {f1_bl:.4f}")
lines.append(report_bl)
lines.append("")
lines.append(f"--- RFECV ({n_sel}d) ---")
lines.append(f"Accuracy = {acc:.4f}   Macro-F1 = {macro_f1:.4f}")
lines.append(report_rf)

with open(os.path.join(OUT_DIR, "table_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ============================================================
# 打印汇总
# ============================================================
p(f"\n{'='*60}")
p(f"Output directory: {OUT_DIR}")
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    sz = os.path.getsize(fp) / 1024 if os.path.isfile(fp) else 0
    p(f"  {fn}  ({sz:.0f} KB)")
p(f"{'='*60}")
p("Done.")
