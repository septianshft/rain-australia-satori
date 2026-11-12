"""
Rain in Australia - Streamlit Deployment
Kelompok 3 "Satori" - Data Science Bootcamp Batch 65 Digital Skola

Jalankan:  streamlit run src/app.py
Membutuhkan artifact hasil training di ../artifacts/ (rain_model.pkl, feature_meta.pkl).
"""
import os
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------- PATH
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACTS = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "rain_model.pkl")
META_PATH = os.path.join(ARTIFACTS, "feature_meta.pkl")

st.set_page_config(page_title="Prediksi Hujan Australia", page_icon="🌦️", layout="wide")


# ---------------------------------------------------------------- LOAD (cached)
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(META_PATH)


MODEL, META = load_artifacts()

# ---------------------------------------------------------------- LABEL & RANGE
LABEL = {
    "MinTemp": "Suhu Minimum (°C)", "MaxTemp": "Suhu Maksimum (°C)",
    "Temp9am": "Suhu jam 9 pagi (°C)", "Temp3pm": "Suhu jam 3 sore (°C)",
    "Rainfall": "Curah Hujan (mm)", "Evaporation": "Evaporasi (mm)",
    "Sunshine": "Sinar Matahari (jam)",
    "WindGustSpeed": "Kecepatan Angin Puncak (km/j)",
    "WindSpeed9am": "Kecepatan Angin 9 pagi (km/j)",
    "WindSpeed3pm": "Kecepatan Angin 3 sore (km/j)",
    "Humidity9am": "Kelembaban 9 pagi (%)", "Humidity3pm": "Kelembaban 3 sore (%)",
    "Pressure9am": "Tekanan Udara 9 pagi (hPa)", "Pressure3pm": "Tekanan Udara 3 sore (hPa)",
    "Cloud9am": "Awan 9 pagi (oktaf)", "Cloud3pm": "Awan 3 sore (oktaf)",
    "Location": "Lokasi", "WindGustDir": "Arah Angin Puncak",
    "WindDir9am": "Arah Angin 9 pagi", "WindDir3pm": "Arah Angin 3 sore",
    "RainToday": "Apakah Hari Ini Hujan?",
}
RANGE = {
    "MinTemp": (-8.5, 34.0, 0.1), "MaxTemp": (-4.8, 48.0, 0.1),
    "Temp9am": (-7.2, 40.0, 0.1), "Temp3pm": (-5.4, 47.0, 0.1),
    "Rainfall": (0.0, 200.0, 0.2), "Evaporation": (0.0, 30.0, 0.1),
    "Sunshine": (0.0, 14.5, 0.1),
    "WindGustSpeed": (6.0, 135.0, 1.0), "WindSpeed9am": (0.0, 100.0, 1.0),
    "WindSpeed3pm": (0.0, 90.0, 1.0),
    "Humidity9am": (0.0, 100.0, 1.0), "Humidity3pm": (0.0, 100.0, 1.0),
    "Pressure9am": (980.0, 1042.0, 0.1), "Pressure3pm": (977.0, 1040.0, 0.1),
    "Cloud9am": (0.0, 9.0, 1.0), "Cloud3pm": (0.0, 9.0, 1.0),
}

# ---------------------------------------------------------------- UI
st.title("🌦️ Prediksi Hujan Besok di Australia")
st.caption("Model: **RainTomorrow** (klasifikasi) · Kelompok 3 Satori · "
           "Data Science Bootcamp Batch 65 Digital Skola")

if MODEL is None:
    st.error("⚠️ Model belum ditemukan. Jalankan dulu `python src/train.py` "
             "untuk melatih & menyimpan model.")
    st.stop()

st.sidebar.header("⚙️ Pengaturan")
st.sidebar.success(f"Model aktif: **{META['model_name']}**\n\n"
                   f"Strategi missing: `{META['missing_strategy']}`\n\n"
                   f"Jumlah fitur: {len(META['num_cols'])} numerik + "
                   f"{len(META['cat_cols'])} kategorikal")

obs_date = st.sidebar.date_input("Tanggal observasi", value=pd.Timestamp("2017-06-01"))
month = obs_date.month
season = (month % 12) // 3
st.sidebar.caption(f"Bulan ke-{month} · Musim (kode) {season} "
                   f"(0=Des-Feb panas, 2=Jun-Agu dingin)")

num_defaults = META["num_defaults"]
cat_options = META["cat_options"]
cat_defaults = META["cat_defaults"]

# Input kategorikal di sidebar (yang paling menentukan)
st.sidebar.subheader("📍 Lokasi & Kondisi Umum")
row = {"Month": month, "Season": season}

loc_idx = cat_options["Location"].index(cat_defaults["Location"]) \
    if cat_defaults["Location"] in cat_options["Location"] else 0
row["Location"] = st.sidebar.selectbox("Lokasi", cat_options["Location"], index=loc_idx)
row["RainToday"] = st.sidebar.selectbox("Hujan hari ini?",
                                        cat_options.get("RainToday", ["No", "Yes"]),
                                        index=0)

# Input numerik & angin di area utama, dikelompokkan
st.subheader("🌡️ Masukkan Kondisi Cuaca Hari Ini")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Suhu**")
    for col in ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]:
        if col in META["num_cols"]:
            lo, hi, st_ = RANGE[col]
            val = min(max(num_defaults[col], lo), hi)
            row[col] = st.slider(LABEL[col], lo, hi, val, st_, key=col)

    st.markdown("**Kelembaban & Awan**")
    for col in ["Humidity9am", "Humidity3pm", "Cloud9am", "Cloud3pm", "Sunshine"]:
        if col in META["num_cols"]:
            lo, hi, st_ = RANGE[col]
            val = min(max(num_defaults[col], lo), hi)
            row[col] = st.slider(LABEL[col], lo, hi, val, st_, key=col)

with c2:
    st.markdown("**Tekanan Udara & Curah Hujan**")
    for col in ["Pressure9am", "Pressure3pm", "Rainfall", "Evaporation"]:
        if col in META["num_cols"]:
            lo, hi, st_ = RANGE[col]
            val = min(max(num_defaults[col], lo), hi)
            row[col] = st.slider(LABEL[col], lo, hi, val, st_, key=col)

    st.markdown("**Angin**")
    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]:
        if col in META["num_cols"]:
            lo, hi, st_ = RANGE[col]
            val = min(max(num_defaults[col], lo), hi)
            row[col] = st.slider(LABEL[col], lo, hi, val, st_, key=col)
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        if col in META["cat_cols"]:
            opts = cat_options[col]
            idx = opts.index(cat_defaults[col]) if cat_defaults[col] in opts else 0
            row[col] = st.selectbox(LABEL[col], opts, index=idx, key=col)

# Fitur numerik/kategorikal lain yang belum tertangani (jaga-jaga kalau kolom berubah)
for col in META["num_cols"]:
    if col not in row and col not in ("Month", "Season"):
        row[col] = st.number_input(LABEL.get(col, col), value=float(num_defaults[col]), key=col)
for col in META["cat_cols"]:
    if col not in row:
        opts = cat_options[col]
        row[col] = st.selectbox(LABEL.get(col, col), opts,
                                index=opts.index(cat_defaults[col]) if cat_defaults[col] in opts else 0,
                                key=col)

# ---------------------------------------------------------------- PREDICT
st.divider()
if st.button("🔮 Prediksi Hujan Besok", type="primary", use_container_width=True):
    data = pd.DataFrame([{c: row[c] for c in META["feature_order"]}])
    prob = float(MODEL.predict_proba(data)[0][1])
    pred = int(prob >= 0.5)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Probabilitas Hujan", f"{prob*100:.1f}%")
        if pred == 1:
            st.error("### ☔ HUJAN besok")
        else:
            st.success("### ☀️ TIDAK hujan besok")
        st.progress(prob)
    with right:
        st.markdown("**Ringkasan input**")
        show = {LABEL.get(k, k): v for k, v in row.items() if k in
                ("Location", "RainToday", "Humidity3pm", "Pressure3pm",
                 "Sunshine", "Rainfall", "Cloud3pm", "WindGustSpeed")}
        st.json(show)

    st.info("💡 **Catatan:** model dilatih dengan *time-based split* dan menangani "
            "imbalance (77% tidak hujan vs 23% hujan). Sinyal terkuat: kelembaban sore, "
            "tekanan udara, sinar matahari, dan apakah hari ini hujan.")
