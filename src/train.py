"""
RAIN IN AUSTRALIA - PREDICTING RainTomorrow (Classification)
Kelompok 3 "Satori" - Data Science Bootcamp Batch 65 Digital Skola

Pipeline: Load -> Clean (replikasi Bagian 6-7 notebook) -> FE -> Time-based Split
          -> Train (LR/RF: impute+scaler+log1p | XGB: NaN native)
          -> Bandingkan -> Simpan model terbaik + metadata.

Jalankan:  python src/train.py
(Untuk Google Colab: lihat blok 'COLAB' di bawah, sesuaikan path CSV.)
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")

# =====================================================================
# KONFIGURASI
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))          # .../src
ROOT = os.path.dirname(HERE)                               # .../Final Project
CSV = os.path.join(ROOT, "weatherAUS.csv")                 # path dataset
ARTIFACTS = os.path.join(ROOT, "artifacts")                # folder model
RESULTS = os.path.join(ROOT, "results")                    # folder hasil
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# COLAB (opsional):
#   from google.colab import drive; drive.mount('/content/drive')
#   CSV = '/content/drive/MyDrive/.../weatherAUS.csv'

# 🔧 KEPUTUSAN TIM - sinkron dengan Data_Understanding.ipynb (satu sumber kebenaran):
#   - Drop Evaporation (missing 43%, sinyal terlemah; benchmark A-E Bagian 6:
#     ROC-AUC 0.8863 vs 0.8866 baseline = tidak rugi, fitur lebih ringan)
#   - Cleaning validitas Bagian 7 direplikasi di apply_validity_cleaning()
#   - log1p(Rainfall) sebagai step pipeline (skew 9.9 -> 2.1; Bagian 8)
#   - XGBoost: NaN native; LR/RF (sklearn): imputasi median/mode di dalam pipeline
DROP_ALWAYS = ["Evaporation"]
TOL_SUHU = 1.0        # derajat C - ambang validitas suhu (Bagian 7.3)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def log(msg):
    print(msg, flush=True)


# =====================================================================
# 1. LOAD, CLEANING & FEATURE ENGINEERING
# =====================================================================
def apply_validity_cleaning(df):
    """Replikasi persis Bagian 7 Data_Understanding.ipynb: nilai mustahil -> NaN.
    Ekspektasi jumlah (142.193 baris): Temp3pm 539, Temp9am 2, Gust 5, Wind9am 1."""
    bad_t3 = ((df["Temp3pm"] < df["MinTemp"] - TOL_SUHU) |
              (df["Temp3pm"] > df["MaxTemp"] + TOL_SUHU))
    bad_t9 = ((df["Temp9am"] < df["MinTemp"] - TOL_SUHU) |
              (df["Temp9am"] > df["MaxTemp"] + TOL_SUHU))
    terisi = df[["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]].notna().all(axis=1)
    bad_gust = terisi & ~(df["WindGustSpeed"] >=
                          df[["WindSpeed9am", "WindSpeed3pm"]].max(axis=1)).fillna(False)
    bad_wind = df["WindSpeed9am"].notna() & ~(df["WindSpeed9am"] <= 100)
    log(f"[clean] suhu luar rentang >{TOL_SUHU}C -> NaN : "
        f"Temp3pm {int(bad_t3.sum())} | Temp9am {int(bad_t9.sum())}")
    log(f"[clean] angin mustahil -> NaN : Gust<wind {int(bad_gust.sum())} | "
        f"WindSpeed9am>100 {int(bad_wind.sum())}")
    df.loc[bad_t3, "Temp3pm"] = np.nan
    df.loc[bad_t9, "Temp9am"] = np.nan
    df.loc[bad_gust, "WindGustSpeed"] = np.nan
    df.loc[bad_wind, "WindSpeed9am"] = np.nan
    return df


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Bagian 6: buang baris tanpa target (tidak bisa dipakai supervised learning)
    before = len(df)
    df = df.dropna(subset=["RainTomorrow"]).copy()
    log(f"[data] baris awal {before:,} -> setelah buang target NaN {len(df):,}")

    # Bagian 7: cleaning validitas (replikasi persis notebook)
    df = apply_validity_cleaning(df)

    # Encode target
    df["y"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})

    # Feature engineering tanggal (musiman). 'Year' TIDAK dipakai sbg fitur
    # agar tidak terjadi ekstrapolasi aneh saat prediksi masa depan.
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(lambda m: (m % 12) // 3)  # 0..3

    # Urutkan berdasarkan waktu untuk time-based split
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    log(f"[split] batas waktu: train <= {df['Date'].iloc[split_idx - 1].date()} < test")

    drop_cols = ["Date", "RainTomorrow"] + [c for c in DROP_ALWAYS if c in df.columns]
    log(f"[config] kolom di-drop: {drop_cols}")
    df = df.drop(columns=drop_cols)
    return df


# =====================================================================
# 2. SPLIT & PREPROCESSOR
# =====================================================================
def build_preprocessor(df, impute=True):
    """impute=True  -> LR/RF (sklearn tak terima NaN): imputer median/mode + scaler.
       impute=False -> XGBoost: NaN dibiarkan native, tanpa imputer/scaler.
       Rainfall selalu lewat log1p sebagai step pipeline (Bagian 8: skew 9.9 -> 2.1),
       sehingga model + transformasi tersimpan dalam satu artifact."""
    cat_cols = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in cat_cols + ["y"] and df[c].dtype != "object"]

    other_num = [c for c in num_cols if c != "Rainfall"]
    log1p = FunctionTransformer(np.log1p)
    if impute:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        rain_pipe = Pipeline([
            ("log", log1p),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = "passthrough"                 # NaN native utk XGBoost
        rain_pipe = Pipeline([("log", log1p)])

    transformers = []
    if "Rainfall" in num_cols:
        transformers.append(("rain", rain_pipe, ["Rainfall"]))
    transformers += [
        ("num", num_pipe, other_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ]
    pre = ColumnTransformer(transformers)
    return pre, num_cols, cat_cols


def time_split(df, test_size=TEST_SIZE):
    split = int(len(df) * (1 - test_size))
    X = df.drop(columns="y")
    y = df["y"]
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# =====================================================================
# 3. MODELS
# =====================================================================
def get_models(pre_dense, pre_native, pos_weight):
    return {
        "LogisticRegression": Pipeline([
            ("pre", pre_dense),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("pre", pre_dense),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("pre", pre_native),
            ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                                  subsample=0.9, colsample_bytree=0.9,
                                  scale_pos_weight=pos_weight, eval_metric="logloss",
                                  tree_method="hist", n_jobs=-1,
                                  random_state=RANDOM_STATE)),
        ]),
    }


# =====================================================================
# 4. MAIN
# =====================================================================
def main():
    t0 = time.time()
    log("=" * 68)
    log("RAIN IN AUSTRALIA - TRAINING & MODEL COMPARISON")
    log("=" * 68)

    df = load_and_prepare(CSV)

    # Bobot untuk imbalance
    pos = int((df["y"] == 1).sum())
    neg = int((df["y"] == 0).sum())
    pos_weight = neg / pos
    log(f"[data] target  No={neg:,} ({neg/len(df)*100:.1f}%)  "
        f"Yes={pos:,} ({pos/len(df)*100:.1f}%)  scale_pos_weight={pos_weight:.2f}")

    X_tr, X_te, y_tr, y_te = time_split(df)
    log(f"[split] time-based  train={len(X_tr):,}  test={len(X_te):,} "
        f"(batas waktu: {df.shape[0]} baris diurutkan by Date)")

    pre_dense, num_cols, cat_cols = build_preprocessor(df, impute=True)
    pre_native, _, _ = build_preprocessor(df, impute=False)
    log(f"[fitur] numeric={len(num_cols)} (Rainfall via log1p)  categorical={len(cat_cols)}")
    log("[pre] LR/RF: impute+scaler | XGBoost: NaN native (keputusan Bagian 6/8)")

    models = get_models(pre_dense, pre_native, pos_weight)
    rows, fitted = [], {}
    for name, model in models.items():
        log(f"\n>>> Training {name} ...")
        ts = time.time()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        dt = time.time() - ts
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_te, pred), 4),
            "Precision": round(precision_score(y_te, pred), 4),
            "Recall": round(recall_score(y_te, pred), 4),
            "F1": round(f1_score(y_te, pred), 4),
            "ROC_AUC": round(roc_auc_score(y_te, proba), 4),
            "TrainSec": round(dt, 1),
        })
        fitted[name] = model
        log(f"    F1={rows[-1]['F1']}  ROC-AUC={rows[-1]['ROC_AUC']}  ({dt:.1f}s)")

    res = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    log("\n" + "=" * 68)
    log("PERBANDINGAN MODEL (urut F1):")
    log("=" * 68)
    log(res.to_string(index=False))

    res_path = os.path.join(RESULTS, "model_comparison.csv")
    res.to_csv(res_path, index=False)
    log(f"\n[save] tabel perbandingan -> {res_path}")

    # Model terbaik
    best_name = res.iloc[0]["Model"]
    best = fitted[best_name]
    log(f"[best] {best_name}  (F1={res.iloc[0]['F1']}, ROC-AUC={res.iloc[0]['ROC_AUC']})")

    pred_te = best.predict(X_te)
    log("\nConfusion Matrix (model terbaik):")
    log(str(confusion_matrix(y_te, pred_te)))
    log("\nClassification Report:")
    log(classification_report(y_te, pred_te, target_names=["No", "Yes"]))

    # ---------------------------------------------------------------- SAVE
    model_path = os.path.join(ARTIFACTS, "rain_model.pkl")
    joblib.dump(best, model_path)
    log(f"[save] model terbaik -> {model_path}")

    # Metadata untuk Streamlit (default input & pilihan kategori)
    meta = {
        "model_name": best_name,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_defaults": {c: float(df[c].median()) for c in num_cols},
        "cat_options": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
        "cat_defaults": {c: df[c].mode(dropna=True).iloc[0] for c in cat_cols},
        "feature_order": num_cols + cat_cols,
        "missing_strategy": "Evaporation dropped; XGB=NaN-native; LR/RF=impute; Rainfall=log1p",
    }
    meta_path = os.path.join(ARTIFACTS, "feature_meta.pkl")
    joblib.dump(meta, meta_path)
    log(f"[save] metadata fitur -> {meta_path}")

    log(f"\n[done] total waktu {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
