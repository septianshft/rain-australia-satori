"""
EDA lengkap - Rain in Australia Prediction (Kel.3 Satori, DS Batch 65)
Menjalankan analisis + menyimpan grafik ke eda/plots/ dan angka kunci ke console.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110

BASE = os.path.dirname(os.path.abspath(__file__))
PLOT = os.path.join(BASE, "plots")
os.makedirs(PLOT, exist_ok=True)
CSV = os.path.join(os.path.dirname(BASE), "weatherAUS.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] plots/{name}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------ LOAD
df = pd.read_csv(CSV)
df["Date"] = pd.to_datetime(df["Date"])

section("1. RINGKASAN DATASET")
print(f"Shape            : {df.shape[0]:,} baris x {df.shape[1]} kolom")
print(f"Rentang tanggal  : {df['Date'].min().date()} -> {df['Date'].max().date()} "
      f"(~{(df['Date'].max()-df['Date'].min()).days/365:.1f} tahun)")
print(f"Jumlah lokasi    : {df['Location'].nunique()}")
print(f"Total missing    : {df.isna().sum().sum():,} sel "
      f"({df.isna().sum().sum()/df.size*100:.1f}% dari data)")
print(f"Baris duplikat   : {df.duplicated().sum():,}")

# ------------------------------------------------------------------ MISSING
section("2. MISSING VALUES")
miss = df.isna().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_tab = pd.DataFrame({"n_missing": miss, "pct": miss_pct})
miss_tab = miss_tab[miss_tab["n_missing"] > 0].sort_values("pct", ascending=False)
print(miss_tab.to_string())

fig, ax = plt.subplots(figsize=(13, 7))
sns.barplot(x=miss_tab["pct"], y=miss_tab.index, ax=ax,
            hue=miss_tab["pct"] > 35, palette={True: "#e74c3c", False: "#3498db"},
            legend=False)
ax.set_xlabel("% Missing")
ax.set_title("Missing Value per Kolom (merah = >35%, kandidat drop)")
save(fig, "01_missing_values.png")

# ------------------------------------------------------------------ TARGET
section("3. TARGET: RainTomorrow")
df_t = df.dropna(subset=["RainTomorrow"])
vc = df_t["RainTomorrow"].value_counts()
vc_pct = df_t["RainTomorrow"].value_counts(normalize=True) * 100
print(f"Target NaN dibuang : {df['RainTomorrow'].isna().sum():,} baris")
print(f"No                 : {vc['No']:,} ({vc_pct['No']:.1f}%)")
print(f"Yes                : {vc['Yes']:,} ({vc_pct['Yes']:.1f}%)")
print(f"Rasio imbalance    : 1 : {vc['No']/vc['Yes']:.2f}")
print(f"Baseline akurasi (selalu prediksi 'No') = {vc_pct['No']:.1f}%")

fig, ax = plt.subplots(figsize=(6, 5))
vc_pct.plot.bar(ax=ax, color=["#3498db", "#e67e22"], width=0.6)
ax.set_ylabel("%")
ax.set_title(f"Target IMBALANCE\n(No {vc_pct['No']:.1f}% vs Yes {vc_pct['Yes']:.1f}%)")
ax.set_xticklabels(["No", "Yes"], rotation=0)
for i, v in enumerate(vc_pct.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
save(fig, "02_target_imbalance.png")

# ------------------------------------------------------------------ TEMPORAL
section("4. POLA TEMPORAL & MUSIMAN")
df_t = df_t.copy()
df_t["Month"] = df_t["Date"].dt.month
df_t["Year"] = df_t["Date"].dt.year
df_t["rain_flag"] = (df_t["RainTomorrow"] == "Yes").astype(int)

monthly = df_t.groupby("Month")["rain_flag"].mean() * 100
print("Probabilitas hujan besok per bulan (%):")
print(monthly.round(1).to_string())
print(f"\nBulan terkering : {monthly.idxmin()} ({monthly.min():.1f}%)")
print(f"Bulan terbasah  : {monthly.idxmax()} ({monthly.max():.1f}%)")

yearly = df_t.groupby("Year")["rain_flag"].mean() * 100
print("\nProbabilitas hujan besok per tahun (%):")
print(yearly.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(monthly.index, monthly.values, marker="o", color="#2980b9", linewidth=2.5)
axes[0].set_title("Musiman: % Hujan per Bulan")
axes[0].set_xlabel("Bulan")
axes[0].set_ylabel("% Hujan")
axes[1].plot(yearly.index, yearly.values, marker="o", color="#27ae60", linewidth=2.5)
axes[1].set_title("Tren Tahunan: % Hujan")
axes[1].set_xlabel("Tahun")
axes[1].set_ylabel("% Hujan")
save(fig, "03_temporal_seasonality.png")

# ------------------------------------------------------------------ LOKASI
section("5. ANALISIS LOKASI")
loc = (df_t.groupby("Location")["rain_flag"].mean() * 100).sort_values()
print("5 lokasi TERING (probabilitas hujan terendah):")
print(loc.head(5).round(1).to_string())
print("\n5 lokasi TERBASAH (probabilitas hujan tertinggi):")
print(loc.tail(5).round(1).to_string())
print(f"\nSebaran antar lokasi: min {loc.min():.1f}% - max {loc.max():.1f}% "
      f"(gap {loc.max()-loc.min():.1f} pp) -> Location PENTING sebagai fitur")

fig, ax = plt.subplots(figsize=(12, 12))
colors = ["#e74c3c" if v < loc.median() else "#3498db" for v in loc.values]
ax.barh(loc.index, loc.values, color=colors)
ax.set_xlabel("% Hujan Besok")
ax.set_title("Probabilitas Hujan per Lokasi (merah=kering, biru=basah)")
save(fig, "04_location.png")

# ------------------------------------------------------------------ NUMERIK vs TARGET
section("6. FITUR NUMERIK vs TARGET (sinyal prediktif)")
num_cols = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
            "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
            "Temp9am", "Temp3pm"]
grp = df_t.groupby("RainTomorrow")[num_cols].mean().T
grp["delta(Yes-No)"] = grp["Yes"] - grp["No"]
grp = grp.sort_values("delta(Yes-No)")
print("Rata-rata fitur saat hujan vs tidak (diurutkan dari beda terbesar):")
print(grp.round(2).to_string())

corr = df_t[num_cols + ["rain_flag"]].corr()["rain_flag"].drop("rain_flag").sort_values()
print("\nKorelasi tiap fitur numerik dengan target hujan (Pearson):")
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#27ae60" if v > 0 else "#c0392b" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Korelasi dengan Hujan Besok")
ax.set_title("Sinyal Prediktif Fitur Numerik (korelasi Pearson)")
save(fig, "05_numeric_correlation.png")

# Top 4 predictor: distribusi per kelas
key_feats = corr.abs().sort_values(ascending=False).head(4).index.tolist()
print(f"\n4 predictor terkuat: {key_feats}")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, f in zip(axes.flat, key_feats):
    sns.histplot(data=df_t, x=f, hue="RainTomorrow", kde=False, bins=40,
                 ax=ax, stat="density", common_norm=False,
                 palette={"No": "#3498db", "Yes": "#e67e22"})
    ax.set_title(f"Distribusi {f} per Kelas")
save(fig, "06_key_predictors_dist.png")

# ------------------------------------------------------------------ KATEGORIK
section("7. FITUR KATEGORIK vs TARGET")
for c in ["RainToday", "WindGustDir"]:
    ct = pd.crosstab(df_t[c], df_t["RainTomorrow"], normalize="index") * 100
    ct = ct.dropna().sort_values("Yes", ascending=False)
    print(f"\n--- {c} : % hujan besok per kategori (top 6) ---")
    print(ct.head(6).round(1).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ct = (pd.crosstab(df_t["RainToday"], df_t["RainTomorrow"], normalize="index")
      * 100).dropna()
ct.plot.bar(ax=ax, color=["#3498db", "#e67e22"], width=0.7)
ax.set_ylabel("% Hujan Besok")
ax.set_title("Pengaruh 'RainToday' terhadap Hujan Besok")
ax.legend(title="RainTomorrow")
save(fig, "07_raintoday_effect.png")

# ------------------------------------------------------------------ KORELASI MATRIX
section("8. HEATMAP KORELASI ANTAR FITUR")
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(df_t[num_cols + ["rain_flag"]].corr(), cmap="coolwarm", center=0,
            annot=False, square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Korelasi Antar Semua Fitur Numerik + Target")
save(fig, "08_corr_heatmap.png")

# ------------------------------------------------------------------ OUTLIER
section("9. CEK KUALITAS DATA / OUTLIER")
print("Statistik deskriptif fitur kunci (perhatikan outlier):")
print(df_t[["Rainfall", "WindGustSpeed", "MaxTemp", "Pressure9am"]]
      .describe().round(1).to_string())
neg_rain = (df_t["Rainfall"] < 0).sum()
print(f"\nRainfall < 0 (tidak valid) : {neg_rain}")
print(f"Rainfall > 200mm (ekstrem)  : {(df_t['Rainfall']>200).sum()} baris")
print("Catatan: ada missing & outlier -> perlu keputusan cleaning tim.")

# ------------------------------------------------------------------ RINGKASAN
section("10. RINGKASAN INSIGHT (buat bahan meeting)")
print(f"""
1. Data besar & time-series: {df.shape[0]:,} baris, ~10 tahun, 49 lokasi.
   -> WAJIB time-based split, jangan random (hindari data leakage).

2. Target IMBALANCE: No {vc_pct['No']:.1f}% vs Yes {vc_pct['Yes']:.1f}% (1:{vc['No']/vc['Yes']:.1f}).
   -> Baseline 'selalu No' = {vc_pct['No']:.1f}%. Jangan pakai Accuracy saja;
      pakai F1/Recall/ROC-AUC + class_weight / scale_pos_weight.

3. Missing value parah di Sunshine (48%), Evaporation (43%),
   Cloud3pm (41%), Cloud9am (38%).
   -> KEPUTUSAN TIM: drop kolom vs imputasi vs eksperimen keduanya.

4. Location SANGAT berpengaruh (gap {loc.max()-loc.min():.0f} pp antar lokasi).
   -> Encoding Location penting (One-Hot / Target Encoding).

5. Predictor terkuat (korelasi): {', '.join(key_feats)}.
   -> Humidity3pm, Pressure3pm, Sunshine, Rainfall dominan.

6. Ada musiman: bulan terbasah {monthly.idxmax()} ({monthly.max():.0f}%),
   terkering {monthly.idxmin()} ({monthly.min():.0f}%).
   -> Ekstrak Month/Season dari Date.

7. RainToday berhubungan kuat dgn RainTomorrow -> kandidat fitur bagus
   (tapi pastikan bukan leakage; ini info hari ini, aman).

8. Ada outlier (Rainfall ekstrem) & nilai negatif -> perlu cleaning.
""")

print("\nSemua grafik tersimpan di:", PLOT)
print("Selesai.")
