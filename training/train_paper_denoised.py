"""
train_paper_denoised.py — 在论文降噪法(db6)重训的 1000 条 paper 数据上训练并评估。

流程: 80/20 分层切分(seed=42) → LinearSVC 全特征 baseline → RFECV(step=100, 5-fold, f1_macro)
标签: 字符串 ["AS","MR","MS","MVP","N"]，与后端 DIAG_INFO 一致，便于直接替换部署模型。
产出:
  out_paper015_dbp/
    exp_data.joblib          完整训练过程数据
    result_summary.json      关键数字汇总
    model_deploy.joblib      部署用模型 dict (含 model/feature_cols/labels/...)
    fig_rfecv_curve.png  fig_confusion.png  fig_roc.png
"""
import os, json, time, warnings, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    classification_report, roc_curve, auc,
)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
PROJECT_DIR = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction"
OUT_DIR = os.path.join(PROJECT_DIR, "out_paper015_dbp")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["AS", "MR", "MS", "MVP", "N"]
COLORS_5 = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
META = {"file", "file_abs", "file_rel", "file_name", "label"}


def p(msg):
    print(msg, flush=True)


def load_paper():
    dfs = []
    for lb in CLASS_NAMES:
        path = os.path.join(PROJECT_DIR, f"extracted_features_{lb}_paper_denoised.csv")
        df = pd.read_csv(path)
        df["label"] = lb
        dfs.append(df)
        p(f"  {lb}: {len(df)} rows")
    return pd.concat(dfs, axis=0, ignore_index=True)


def build_xy(df):
    fc = sorted([str(c) for c in df.columns if c not in META])
    X = df[fc].to_numpy(dtype=np.float64)
    y = np.array(df["label"].tolist())
    return X, y, fc


def main():
    p("=" * 70)
    p("Train: paper-only 1000 (db6 denoised) -> LinearSVC baseline + RFECV")
    p("=" * 70)

    df = load_paper()
    X, y, fc = build_xy(df)
    p(f"Total: {X.shape[0]} samples, {X.shape[1]} features")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    p(f"Split: train={Xtr.shape[0]}, test={Xte.shape[0]}")

    # ---- baseline ----
    t0 = time.time()
    bl = Pipeline([("scaler", StandardScaler()),
                   ("svm", LinearSVC(C=1.0, dual="auto", random_state=RANDOM_SEED, max_iter=2000))])
    bl.fit(Xtr, ytr)
    yp_bl = bl.predict(Xte)
    bl_acc = accuracy_score(yte, yp_bl)
    bl_f1 = f1_score(yte, yp_bl, average="macro")
    p(f"Baseline(6373d): Acc={bl_acc:.4f}, macro-F1={bl_f1:.4f} ({time.time()-t0:.0f}s)")

    # ---- RFECV ----
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(estimator=LinearSVC(dual="auto", random_state=RANDOM_SEED, max_iter=2000),
                  step=100, cv=cv, scoring="f1_macro", n_jobs=1)
    clf = Pipeline([("scaler", StandardScaler()), ("rfecv", rfecv)])
    clf.fit(Xtr, ytr)
    r = clf.named_steps["rfecv"]
    n_feat = int(r.n_features_)
    cv_scores = r.cv_results_["mean_test_score"]
    cv_std = r.cv_results_["std_test_score"]
    n_list = r.cv_results_["n_features"]
    best_cv = float(cv_scores[np.argmax(cv_scores)])
    p(f"RFECV: optimal={n_feat} features, CV macro-F1={best_cv:.4f} ({time.time()-t0:.0f}s)")

    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    prec_m = precision_score(yte, yp, average="macro")
    rec_m = recall_score(yte, yp, average="macro")
    report = classification_report(yte, yp, target_names=CLASS_NAMES, digits=4)
    p(f"Test: Acc={acc:.4f}, macro-F1={f1m:.4f}, macro-P={prec_m:.4f}, macro-R={rec_m:.4f}")

    # ---- ROC (one-vs-rest, decision_function) ----
    Xs = clf.named_steps["scaler"].transform(Xte)
    Xs = clf.named_steps["rfecv"].transform(Xs)
    ds = clf.named_steps["rfecv"].estimator_.decision_function(Xs)
    yb = label_binarize(yte, classes=CLASS_NAMES)
    roc_aucs = {}
    if ds.ndim == 1:
        ds = np.column_stack([-ds, ds])
    for i, lb in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(yb[:, i], ds[:, i])
        roc_aucs[lb] = float(auc(fpr, tpr))

    # ---- per-class ----
    per_class = {}
    for i, lb in enumerate(CLASS_NAMES):
        per_class[lb] = {
            "precision": float(precision_score(yte, yp, labels=[lb], average=None)[0]),
            "recall": float(recall_score(yte, yp, labels=[lb], average=None)[0]),
            "f1": float(f1_score(yte, yp, labels=[lb], average=None)[0]),
            "auc": roc_aucs[lb],
        }

    selected_names = [fc[i] for i in range(len(fc)) if r.support_[i]]

    result_summary = {
        "denoising": "paper: db6 wavelet + BayesShrink + Butterworth 20-500Hz + 65/35 mix + RMS norm",
        "protocol": "80/20 stratified split (seed=42), LinearSVC C=1.0, RFECV step=100, 5-fold CV, scoring=f1_macro",
        "n_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "n_selected": n_feat,
        "cv_macro_f1": best_cv,
        "baseline": {"acc": float(bl_acc), "f1_macro": float(bl_f1)},
        "test": {
            "acc": float(acc), "f1_macro": float(f1m),
            "precision_macro": float(prec_m), "recall_macro": float(rec_m),
        },
        "per_class": per_class,
        "classification_report": report,
    }

    # ---- saved artifacts ----
    exp_data = {
        "X": X, "y": y, "fc": fc,
        "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
        "baseline": {"acc": bl_acc, "f1": bl_f1, "pipeline": bl, "y_pred": yp_bl},
        "rfecv": {
            "pipeline": clf, "n": n_feat, "support": r.support_, "ranking": r.ranking_,
            "cv_scores": cv_scores, "cv_std": cv_std, "n_list": n_list, "optimal_score": best_cv,
        },
        "eval": {"acc": acc, "f1": f1m, "y_pred": yp, "report": report, "per_class": per_class},
        "selected_features": selected_names,
    }
    joblib.dump(exp_data, os.path.join(OUT_DIR, "exp_data.joblib"))
    with open(os.path.join(OUT_DIR, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2, default=str)

    # deployment model dict (matches backend engine.py expectations)
    model_deploy = {
        "model": clf,                       # Pipeline(scaler + RFECV), predict -> string label
        "feature_cols": fc,                 # full 6373, training order
        "labels": CLASS_NAMES,
        "feature_count": int(X.shape[1]),
        "n_selected_features": n_feat,
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "denoising": result_summary["denoising"],
        "trained_on": "paper 1000 recordings, db6-denoised",
    }
    joblib.dump(model_deploy, os.path.join(OUT_DIR, "model_deploy.joblib"))
    p("Saved: exp_data.joblib / result_summary.json / model_deploy.joblib")

    # ---- figures ----
    # 1) RFECV curve
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(n_list, cv_scores, color="#0072B2", lw=1.8, label="CV mean macro-F1")
    ax.fill_between(n_list, cv_scores - cv_std, cv_scores + cv_std, color="#0072B2", alpha=0.15)
    ax.axvline(n_feat, color="#D55E00", ls="--", lw=1.5, label=f"Optimal = {n_feat}")
    ax.scatter([n_feat], [best_cv], color="#D55E00", s=80, zorder=5)
    ax.set_xlabel("Number of features selected", fontsize=12)
    ax.set_ylabel("5-fold CV macro F1", fontsize=12)
    ax.set_title("RFECV Feature Selection (db6-denoised, 1000 recordings)", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xlim(max(n_list), min(n_list))
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_rfecv_curve.png"), dpi=200); plt.close(fig)

    # 2) confusion matrix
    cm = confusion_matrix(yte, yp, labels=CLASS_NAMES)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cmn, dtype=object)
    for i in range(5):
        for j in range(5):
            annot[i, j] = f"{cm[i, j]}\n{cmn[i, j]:.1%}"
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=10, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (Acc={acc:.3f})", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_confusion.png"), dpi=200); plt.close(fig)

    # 3) ROC
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for i, (lb, c) in enumerate(zip(CLASS_NAMES, COLORS_5)):
        fpr, tpr, _ = roc_curve(yb[:, i], ds[:, i])
        ax.plot(fpr, tpr, color=c, lw=2.0, label=f"{lb} (AUC={roc_aucs[lb]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("ROC (db6-denoised, 1000 recordings)", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_roc.png"), dpi=200); plt.close(fig)

    p("\n[DONE]")
    p(f"  Baseline  : Acc={bl_acc:.4f}, macro-F1={bl_f1:.4f}")
    p(f"  RFECV     : {n_feat} features, CV-F1={best_cv:.4f}, Test Acc={acc:.4f}, macro-F1={f1m:.4f}")
    p(f"  per-class : " + ", ".join(f"{k}={per_class[k]['f1']:.3f}" for k in CLASS_NAMES))


if __name__ == "__main__":
    main()