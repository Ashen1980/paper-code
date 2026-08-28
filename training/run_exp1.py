"""Exp1: RFECV on paper-only 1000 Denoised + all figures"""
import sys, os, time, warnings, joblib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
PROJECT_DIR = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction"
OUT_DIR = os.path.join(PROJECT_DIR, "out_paper015_rfecv")
os.makedirs(OUT_DIR, exist_ok=True)

PAPER_LABELS = ["AS", "MR", "MS", "MVP", "N"]
NUM_IDX = {"AS": 0, "MR": 1, "MS": 2, "MVP": 3, "N": 4}
CLASS_NAMES = ["AS", "MR", "MS", "MVP", "N"]
COLORS_5 = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

def p(msg): print(msg, flush=True)

def load_paper(suffix=""):
    dfs = []
    for lb in PAPER_LABELS:
        path = os.path.join(PROJECT_DIR, f"extracted_features_{lb}{suffix}.csv")
        df = pd.read_csv(path); df["label"] = lb; dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)

def load_synth(path):
    df = pd.read_csv(path); df["label"] = "N"; return df

def build_xy(df):
    meta = {"file", "file_abs", "file_rel", "file_name", "label"}
    fc = sorted([str(c) for c in df.columns if c not in meta])
    X = df[fc].to_numpy(dtype=np.float32)
    y = np.array([NUM_IDX[lb] for lb in df["label"]], dtype=np.int32)
    return X, y, fc

try:
    p("="*60)
    p("EXP1: Paper-only 1000 Denoised → RFECV")
    p("="*60)

    # Load
    p("\n[1] Loading data...")
    paper_dn = load_paper("_paper_denoised")
    p(f"  Loaded: {len(paper_dn)} samples, {len(paper_dn.columns)} columns")

    X, y, fc = build_xy(paper_dn)
    p(f"  Features: {X.shape}")

    # Split
    p("\n[2] Train/test split...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    p(f"  Train={Xtr.shape}, Test={Xte.shape}")

    # Baseline
    p("\n[3] Baseline LinearSVC (6373d)...")
    t0 = time.time()
    clf_bl = Pipeline([("sc", StandardScaler()),
                       ("svm", LinearSVC(C=1.0, dual="auto", random_state=RANDOM_SEED, max_iter=2000))])
    clf_bl.fit(Xtr, ytr)
    yp_bl = clf_bl.predict(Xte)
    bl_acc = accuracy_score(yte, yp_bl)
    bl_f1 = f1_score(yte, yp_bl, average="macro")
    p(f"  Baseline: Acc={bl_acc:.4f}, F1_macro={bl_f1:.4f} ({time.time()-t0:.0f}s)")

    # RFECV
    p("\n[4] RFECV (step=100, 5-fold CV)...")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(estimator=LinearSVC(dual="auto", random_state=RANDOM_SEED, max_iter=2000),
                  step=100, cv=cv, scoring="f1_macro", n_jobs=1, verbose=0)
    clf = Pipeline([("sc", StandardScaler()), ("rfecv", rfecv)])
    clf.fit(Xtr, ytr)
    r = clf.named_steps["rfecv"]
    n_feat = r.n_features_
    opt_idx = np.argmax(r.cv_results_["mean_test_score"])
    best_cv = r.cv_results_["mean_test_score"][opt_idx]
    p(f"  Optimal: {n_feat} features, CV F1={best_cv:.4f} ({time.time()-t0:.0f}s)")

    # Eval
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    report = classification_report(yte, yp, target_names=CLASS_NAMES, digits=4)
    p(f"  Test: Acc={acc:.4f}, F1_macro={f1m:.4f}")
    p(report)

    # Save
    save = {"pipeline": clf, "n_selected": n_feat, "feature_cols": fc,
            "support": r.support_, "ranking": r.ranking_,
            "eval": {"acc": acc, "f1": f1m, "y_pred": yp, "report": report}}
    save_j = {"pipeline": clf, "n_selected": n_feat, "feature_cols": fc,
              "support": r.support_, "ranking": r.ranking_,
              "eval": {"acc": acc, "f1": f1m, "y_pred": yp, "report": report}}
    joblib.dump(save_j, os.path.join(OUT_DIR, "rfecv_result_exp1.joblib"))
    p(f"  Saved: rfecv_result_exp1.joblib")

    # Also save full dict for figure generation
    exp1_data = {"X": X, "y": y, "fc": fc, "X_train": Xtr, "y_train": ytr,
                 "X_test": Xte, "y_test": yte,
                 "bl": {"acc": bl_acc, "f1": bl_f1, "y_pred": yp_bl, "clf": clf_bl},
                 "rfecv": {"pipeline": clf, "n": n_feat, "support": r.support_, "ranking": r.ranking_,
                           "cv_scores": r.cv_results_["mean_test_score"],
                           "cv_std": r.cv_results_["std_test_score"],
                           "n_list": r.cv_results_["n_features"],
                           "optimal_score": best_cv},
                 "eval": {"acc": acc, "f1": f1m, "y_pred": yp, "report": report},
                 "name": "Exp1: Paper-only"}
    joblib.dump(exp1_data, os.path.join(OUT_DIR, "exp1_data_full.joblib"))

    # ── Figures ──
    p("\n[5] Generating figures...")

    # RFECV curve
    n_list = list(r.cv_results_["n_features"])[::-1]
    scores = list(r.cv_results_["mean_test_score"])[::-1]
    stds = list(r.cv_results_["std_test_score"])[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(n_list, [s-t for s,t in zip(scores,stds)], [s+t for s,t in zip(scores,stds)],
                    alpha=0.18, color="#0072B2")
    ax.plot(n_list, scores, color="#0072B2", lw=1.8, label="CV Mean F1 (macro)")
    ax.axvline(x=n_feat, color="#D55E00", ls="--", lw=1.5, label=f"Optimal: {n_feat}d (F1={best_cv:.4f})")
    ax.scatter([n_feat], [best_cv], color="#D55E00", s=80, zorder=5)
    ax.set_xlabel("N Features"); ax.set_ylabel("5-Fold CV Macro F1")
    ax.set_title("RFECV \u2014 Exp1: Paper-only (1000)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.25); ax.set_xlim(max(n_list), min(n_list))
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_rfecv_curve_exp1.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    p("  fig_rfecv_curve_exp1.png")

    # Confusion Matrix
    cm = confusion_matrix(yte, yp, labels=range(5))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(5):
        for j in range(5): annot[i,j] = f"{cm[i,j]}\n{cm_norm[i,j]:.1%}"
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    hm = sns.heatmap(cm_norm, annot=annot, fmt="", cmap=plt.cm.Blues,
                     vmin=0, vmax=1, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                     linewidths=1.5, linecolor="white",
                     annot_kws={"fontsize": 10, "fontweight": "normal"},
                     cbar_kws={"label": "Recall", "shrink": 0.82})
    for i in range(5):
        tick = hm.texts[i*5+i]; tick.set_fontweight("bold"); tick.set_fontsize(11)
    ax.set_title("Exp1: Paper-only (1000)", fontsize=12, fontweight="normal", pad=10)
    ax.set_xlabel("Predicted label", fontsize=11); ax.set_ylabel("True label", fontsize=11)
    ax.tick_params(labelsize=10)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_cm_exp1.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    p("  fig_cm_exp1.png")

    # ROC
    Xs = clf.named_steps["sc"].transform(Xte)
    Xs = clf.named_steps["rfecv"].transform(Xs)
    ds = clf.named_steps["rfecv"].estimator_.decision_function(Xs)
    yb = label_binarize(yte, classes=range(5))
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for i, (lb, c) in enumerate(zip(CLASS_NAMES, COLORS_5)):
        fpr, tpr, _ = roc_curve(yb[:, i], ds[:, i])
        ax.plot(fpr, tpr, color=c, lw=2.0, label=f"{lb} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1.0,alpha=0.4); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False positive rate", fontsize=11); ax.set_ylabel("True positive rate", fontsize=11)
    ax.legend(fontsize=9, frameon=True, edgecolor="lightgray"); ax.grid(alpha=0.15)
    ax.set_title("ROC \u2014 Exp1: Paper-only (1000)", fontsize=12, fontweight="normal", pad=10)
    ax.tick_params(labelsize=10)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_roc_exp1.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    p("  fig_roc_exp1.png")

    # Feature Pie
    sel_names = [fc[i] for i in range(len(fc)) if r.support_[i]]
    cats = {"MFCC": 0, "audSpec_Rfilt": 0, "audSpec": 0, "pcm_fftMag": 0,
            "pcm_zcr": 0, "pcm_RMSenergy": 0, "F0/voicing": 0,
            "jitter/shimmer": 0, "loudness": 0, "Other": 0}
    for name in sel_names:
        if "mfcc" in name: cats["MFCC"] += 1
        elif "audSpec_Rfilt" in name: cats["audSpec_Rfilt"] += 1
        elif "audSpec" in name: cats["audSpec"] += 1
        elif "pcm_fftMag" in name: cats["pcm_fftMag"] += 1
        elif "pcm_zcr" in name: cats["pcm_zcr"] += 1
        elif "pcm_RMSenergy" in name: cats["pcm_RMSenergy"] += 1
        elif "F0" in name or "voicing" in name or "logHNR" in name: cats["F0/voicing"] += 1
        elif "jitter" in name or "shimmer" in name: cats["jitter/shimmer"] += 1
        elif "loudness" in name.lower(): cats["loudness"] += 1
        else: cats["Other"] += 1
    cats = {k: v for k, v in cats.items() if v > 0}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(cats.values(), labels=cats.keys(), autopct="%1.1f%%",
           colors=plt.cm.Set2(np.linspace(0, 1, len(cats))), startangle=140)
    ax.set_title(f"Feature Composition \u2014 Exp1: Paper-only ({n_feat} features)", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_feature_pie_exp1.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    p("  fig_feature_pie_exp1.png")

    p(f"\n[DONE] Exp1 complete! Features: 6373 -> {n_feat}, Acc: {acc:.4f}")

except Exception as e:
    import traceback
    traceback.print_exc()
    p(f"ERROR: {e}")
    sys.exit(1)
