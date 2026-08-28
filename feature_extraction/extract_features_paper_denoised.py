"""
extract_features_paper_denoised.py — 提取降噪后paper数据的ComParE_2016特征

输入:  001_paper015/{AS,MR,MS,MVP,N}_denoised/*.wav
输出:  extracted_features_{AS,MR,MS,MVP,N}_paper_denoised.csv
"""

import os, tempfile, numpy as np, pandas as pd
import opensmile, librosa, soundfile as sf
from pathlib import Path

PROJECT_DIR = r"c:\Users\26287\Desktop\Work\Feature Extraction2.0\Feature Extraction"
DATA_DIR = os.path.join(PROJECT_DIR, "001_paper015")
CATEGORIES = ["AS", "MR", "MS", "MVP", "N"]

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

print("=" * 80)
print("Paper 降噪数据 OpenSMILE 特征提取")
print("=" * 80)

def extract_folder(folder_path, output_csv, tag):
    wavs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
    total = len(wavs)
    if total == 0: print(f"  [{tag}] 无文件"); return
    print(f"  [{tag}] {total} files → {os.path.basename(output_csv)}")
    all_features = []
    for i, fn in enumerate(wavs, 1):
        if i % 50 == 0: print(f"    {i}/{total}")
        fp = os.path.join(folder_path, fn)
        try:
            y, _ = librosa.load(fp, sr=8000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, y, 8000)
            fdf = smile.process_file(tmp_path)
            os.unlink(tmp_path)
            fdf = fdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            d = fdf.iloc[0].to_dict()
            d["file"] = os.path.abspath(fp)
            all_features.append(d)
        except Exception as e:
            print(f"    ERR {fn}: {e}")
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"  [{tag}] OK: {len(df)} samples, {len(df.columns)-1} features")

for cat in CATEGORIES:
    folder = os.path.join(DATA_DIR, f"{cat}_denoised")
    csv_path = os.path.join(PROJECT_DIR, f"extracted_features_{cat}_paper_denoised.csv")
    if not os.path.isdir(folder):
        print(f"  [{cat}] skip: no denoised dir")
        continue
    extract_folder(folder, csv_path, cat)

print("\nDone!")
