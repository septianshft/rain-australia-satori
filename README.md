# 🌦️ Rain in Australia — Prediksi `RainTomorrow`

Final Project Data Science Bootcamp **Batch 65 — Digital Skola** · Kelompok 3 **"Satori"**

Model klasifikasi untuk memprediksi apakah besok hujan di Australia berdasarkan observasi
cuaca harian (dataset [weatherAUS](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package), 145.460 baris, 49 lokasi, 2007–2017).

## Alur Proyek (CRISP-DM)

| Tahap | File |
|---|---|
| Data Understanding + Preparation (EDA, cleaning, feature engineering) | [`Data_Understanding.ipynb`](Data_Understanding.ipynb) |
| Modeling + Evaluasi (LR vs RF vs XGBoost) | [`Rain_Australia_Final_Project.ipynb`](Rain_Australia_Final_Project.ipynb), [`src/train.py`](src/train.py) |
| Deployment (Streamlit) | [`src/app.py`](src/app.py) |

**Model terbaik: XGBoost** — ROC-AUC 0.887, F1 0.657 (lihat `results/model_comparison.csv`).

## Struktur Folder

```
├── Data_Understanding.ipynb        # notebook utama (mentoring & presentasi)
├── Rain_Australia_Final_Project.ipynb  # notebook modeling (Colab)
├── weatherAUS.csv                  # dataset
├── src/
│   ├── app.py                      # aplikasi Streamlit
│   └── train.py                    # pipeline training resmi → results/ + artifacts/
├── artifacts/                      # rain_model.pkl + feature_meta.pkl (dipakai app)
├── results/                        # CSV perbandingan model
├── eda/                            # skrip EDA standalone + plots PNG
└── requirements.txt                # versi di-pin (wajib untuk kompatibilitas pickle)
```

## Menjalankan

```bash
pip install -r requirements.txt

# Training ulang (menghasilkan artifacts/ dan results/)
python src/train.py

# Aplikasi Streamlit lokal
streamlit run src/app.py
```

Notebook bisa dibuka langsung di **Google Colab** (File → Open from GitHub); jika
`weatherAUS.csv` tidak ada di lingkungan Colab, sel loader akan mengunduhnya otomatis
dari repo ini.

## Deployment

Aplikasi di-deploy ke **Streamlit Community Cloud**:

- Repo: `septianshft/rain-australia-satori` (branch `main`)
- Main file: `src/app.py`
- Setiap push ke `main` otomatis men-deploy ulang aplikasi.

## Keputusan Kunci

- **Split time-based 80/20** (tanpa shuffle) — data time-series, hindari leakage.
- **Drop `Evaporation`** (missing 43%) — benchmark A–E di Bagian 6: performa setara baseline, fitur lebih ringan.
- **NaN dibiarkan** untuk XGBoost (native handling); imputasi hanya di dalam pipeline LR/RF.
- **Class imbalance** (77.6% No) ditangani dengan `scale_pos_weight`, bukan SMOTE.
- Detail lengkap + decision log ada di `Data_Understanding.ipynb`.
