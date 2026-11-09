import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CompuPrice | AI Price Estimator",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "computer_prices.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_NAME_PATH = os.path.join(BASE_DIR, "models", "best_model_name.txt")

# =========================================================
# CUSTOM STYLE
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #172b4d;
        --muted: #65758b;
        --blue: #174A8B;
        --blue-deep: #103665;
        --blue-soft: #eaf1fa;
        --gold: #F2C94C;
        --line: #e3eaf3;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 92% 2%, rgba(242,201,76,.13), transparent 23rem),
            #f6f8fb;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(246,248,251,.92); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #233b5d; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: rgb(0, 84, 163); }
    .block-container { max-width: 1220px; padding-top: 2.1rem; padding-bottom: 3rem; }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
        color: var(--ink);
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.35rem 2.5rem;
        border-radius: 24px;
        color: #ffffff;
        background: linear-gradient(115deg, #103665 0%, #174A8B 68%, #2864a6 100%);
        box-shadow: 0 12px 30px rgba(16,54,101,.16);
        margin-bottom: 1.5rem;
    }
    .hero:before {
        content: "";
        position: absolute;
        width: 260px; height: 260px;
        right: -65px; top: -125px;
        border: 2px solid rgba(242,201,76,.45);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(255,255,255,.045),
                    0 0 0 52px rgba(255,255,255,.035);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 90px; height: 6px;
        left: 2.5rem; bottom: 1.1rem;
        border-radius: 10px;
        background: var(--gold);
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .72rem;
        font-weight: 800;
        color: var(--gold);
        margin-bottom: .7rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.08;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 .75rem 0;
    }
    .hero-copy {
        color: #e3edf9;
        font-size: 1rem;
        max-width: 650px;
        line-height: 1.7;
        margin: 0;
    }
    .hero-chip {
        display: inline-block;
        margin-top: 1.25rem;
        padding: .48rem .85rem;
        border-radius: 999px;
        background: rgba(242,201,76,.16);
        border: 1px solid rgba(242,201,76,.65);
        font-size: .78rem;
        font-weight: 700;
        color: #fff4c2;
    }
    .section-kicker {
        color: #174A8B;
        text-transform: uppercase;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        margin: .2rem 0 .25rem 0;
    }
    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: #202f49;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0 0 .25rem 0;
    }
    .section-description { color: #718096; font-size: .91rem; margin-bottom: 1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 7px 22px rgba(25,55,90,.045);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-top: 3px solid var(--blue);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 18px rgba(25,55,90,.04);
    }
    div[data-testid="stMetricLabel"] { color: #718096; }
    div[data-testid="stMetricValue"] {
        color: #174A8B;
        font-family: 'Space Grotesk', sans-serif;
    }
    .result-card {
        background: linear-gradient(110deg, #eaf1fa 0%, #f5f8fc 100%);
        border: 1px solid #cbd9eb;
        border-left: 6px solid #174A8B;
        border-radius: 20px;
        padding: 1.65rem 1.8rem;
        color: #172b4d;
        box-shadow: 0 9px 24px rgba(23,74,139,.08);
    }
    .result-label {
        color: #174A8B;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-weight: 800;
    }
    .result-price {
        color: #103665;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.1rem, 5vw, 3.2rem);
        line-height: 1.2;
        font-weight: 700;
        margin: .4rem 0;
    }
    .result-note { color: #5d6f86; font-size: .88rem; }
    .small-note { color: #7c879b; font-size: .8rem; line-height: 1.6; }
    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        padding: .78rem 1rem;
        color: #ffffff;
        font-weight: 800;
        background: #174A8B;
        box-shadow: 0 7px 16px rgba(23,74,139,.20);
        transition: .2s ease;
    }
    .stButton > button:hover {
        color: #ffffff;
        background: #103665;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(23,74,139,.25);
    }
    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
    }
    hr { border-color: #e3eaf3; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = load_dataset()
model = load_model()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: .7rem .2rem 1.2rem .2rem;">
            <div style="font-size:2rem;">✦</div>
            <div style="font-family:'Space Grotesk';font-size:1.35rem;font-weight:700;">
                CompuPrice
            </div>
            <div style="font-size:.78rem;color:#68758a;">SMART COMPUTER PRICE ESTIMATOR</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**NAVIGASI**")
    page = st.radio(
        "Pilih halaman",
        ["Price Predictor", "About Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**MODEL INFORMATION**")
    st.markdown("🧠 LightGBM Regressor")
    st.markdown("📊 Dataset: Computer Prices 2025")
    st.markdown("📦 100.000 data komputer")
    st.markdown("---")
    st.caption("Machine Learning Practicum • Kelompok 5")
    st.caption("Universitas Airlangga")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered price estimation</div>
        <div class="hero-title">Find the right value<br>for your next computer.</div>
        <p class="hero-copy">
            Masukkan spesifikasi komputer atau laptop untuk mendapatkan estimasi harga
            berdasarkan model Machine Learning yang telah dilatih menggunakan dataset
            Computer Prices 2025.
        </p>
        <span class="hero-chip">✦ Powered by LightGBM Regressor</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model belum ditemukan. Pastikan file "
        "`models/best_model.joblib` tersedia di folder proyek."
    )
    st.stop()

if df is None:
    st.error(
        "Dataset `computer_prices.csv` tidak ditemukan. Letakkan file tersebut "
        "sejajar dengan `app.py` agar pilihan spesifikasi dapat dimuat."
    )
    st.stop()

# Features used when training: model, cpu_model, and price were excluded.
feature_columns = [c for c in df.columns if c not in ["model", "cpu_model", "price"]]
numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df[feature_columns].select_dtypes(exclude=[np.number]).columns.tolist()


def options_for(column):
    values = df[column].dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return values


def numeric_default(column):
    value = df[column].median()
    if pd.isna(value):
        return 0
    return float(value)


def number_input_for(column, label, help_text=None):
    minimum = float(df[column].min())
    maximum = float(df[column].max())
    default = numeric_default(column)
    if minimum == maximum:
        return st.number_input(label, value=default, help=help_text)
    # Step sizes are chosen to make common hardware values convenient to enter.
    integer_like = pd.api.types.is_integer_dtype(df[column])
    step = 1.0 if integer_like else 0.1
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=min(max(default, minimum), maximum),
        step=step,
        help=help_text,
    )


if page == "Price Predictor":
    st.markdown('<div class="section-kicker">01 / Configure your device</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Computer specifications</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Lengkapi spesifikasi perangkat. Nilai awal diambil dari nilai tengah dataset.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        # Device
        with st.container(border=True):
            st.markdown("#### 🖥️ Device overview")
            c1, c2, c3 = st.columns(3)
            values = {}
            with c1:
                col = "device_type"
                opts = options_for(col)
                values[col] = st.selectbox("Device type", opts, index=0)
            with c2:
                col = "brand"
                opts = options_for(col)
                values[col] = st.selectbox("Brand", opts)
            with c3:
                col = "release_year"
                values[col] = number_input_for(col, "Release year")

            c1, c2, c3 = st.columns(3)
            with c1:
                col = "os"
                values[col] = st.selectbox("Operating system", options_for(col))
            with c2:
                col = "form_factor"
                values[col] = st.selectbox("Form factor", options_for(col))
            with c3:
                col = "warranty_months"
                values[col] = number_input_for(col, "Warranty (months)")

        # CPU and GPU
        with st.container(border=True):
            st.markdown("#### ⚡ Processor & graphics")
            c1, c2, c3 = st.columns(3)
            with c1:
                col = "cpu_brand"
                values[col] = st.selectbox("CPU brand", options_for(col))
            with c2:
                col = "cpu_tier"
                values[col] = number_input_for(col, "CPU tier")
            with c3:
                col = "cpu_cores"
                values[col] = number_input_for(col, "CPU cores")

            c1, c2, c3 = st.columns(3)
            with c1:
                col = "cpu_threads"
                values[col] = number_input_for(col, "CPU threads")
            with c2:
                col = "cpu_base_ghz"
                values[col] = number_input_for(col, "CPU base clock (GHz)")
            with c3:
                col = "cpu_boost_ghz"
                values[col] = number_input_for(col, "CPU boost clock (GHz)")

            c1, c2, c3 = st.columns(3)
            with c1:
                col = "gpu_brand"
                values[col] = st.selectbox("GPU brand", options_for(col))
            with c2:
                col = "gpu_model"
                values[col] = st.selectbox("GPU model", options_for(col))
            with c3:
                col = "gpu_tier"
                values[col] = number_input_for(col, "GPU tier")

            c1, c2 = st.columns(2)
            with c1:
                col = "vram_gb"
                values[col] = number_input_for(col, "VRAM (GB)")
            with c2:
                col = "psu_watts"
                values[col] = number_input_for(col, "PSU power (W)")

        # Memory and storage
        with st.container(border=True):
            st.markdown("#### 💾 Memory & storage")
            c1, c2, c3 = st.columns(3)
            with c1:
                col = "ram_gb"
                values[col] = number_input_for(col, "RAM (GB)")
            with c2:
                col = "storage_type"
                values[col] = st.selectbox("Storage type", options_for(col))
            with c3:
                col = "storage_gb"
                values[col] = number_input_for(col, "Storage capacity (GB)")

            c1, c2 = st.columns(2)
            with c1:
                col = "storage_drive_count"
                values[col] = number_input_for(col, "Number of storage drives")
            with c2:
                col = "weight_kg"
                values[col] = number_input_for(col, "Device weight (kg)")

        # Display and connectivity
        with st.container(border=True):
            st.markdown("#### 🖼️ Display, battery & connectivity")
            c1, c2, c3 = st.columns(3)
            with c1:
                col = "display_type"
                values[col] = st.selectbox("Display type", options_for(col))
            with c2:
                col = "display_size_in"
                values[col] = number_input_for(col, "Display size (inch)")
            with c3:
                col = "resolution"
                values[col] = st.selectbox("Resolution", options_for(col))

            c1, c2, c3 = st.columns(3)
            with c1:
                col = "refresh_hz"
                values[col] = number_input_for(col, "Refresh rate (Hz)")
            with c2:
                col = "battery_wh"
                values[col] = number_input_for(col, "Battery capacity (Wh)")
            with c3:
                col = "charger_watts"
                values[col] = number_input_for(col, "Charger power (W)")

            c1, c2 = st.columns(2)
            with c1:
                col = "wifi"
                values[col] = st.selectbox("Wi-Fi standard", options_for(col))
            with c2:
                col = "bluetooth"
                values[col] = number_input_for(col, "Bluetooth version")

        submitted = st.form_submit_button("✦  Estimate computer price")

    if submitted:
        # Ensure the input columns and order exactly match the training features.
        input_df = pd.DataFrame([{col: values[col] for col in feature_columns}])
        for col in numeric_columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        try:
            prediction = float(model.predict(input_df)[0])
            st.markdown("")
            st.markdown('<div class="section-kicker">02 / Prediction result</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">Your estimated price</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated computer price</div>
                    <div class="result-price">${prediction:,.2f}</div>
                    <div class="result-note">Estimated by LightGBM Regressor · Currency: USD</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Hasil merupakan estimasi model berdasarkan pola pada dataset, "
                "bukan harga jual resmi atau penawaran toko."
            )
            with st.expander("Lihat kembali spesifikasi yang digunakan"):
                st.dataframe(input_df.T.rename(columns={0: "Input"}), use_container_width=True)
        except Exception as exc:
            st.error(f"Prediksi gagal dilakukan: {exc}")
            st.info(
                "Pastikan semua fitur input sesuai dengan fitur saat training. "
                "Jika error berlanjut, kirimkan pesan error lengkapnya."
            )

else:
    st.markdown('<div class="section-kicker">PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">About CompuPrice</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">A machine learning project for estimating computer and laptop prices.</div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Dataset records", f"{len(df):,}")
    b.metric("Input features", len(feature_columns))
    c.metric("Selected model", "LightGBM")
    with st.container(border=True):
        st.markdown("### 🎯 Project objective")
        st.write(
            "CompuPrice membantu memperkirakan harga komputer dan laptop berdasarkan "
            "spesifikasi perangkat keras, perangkat lunak, dan karakteristik perangkat."
        )
        st.markdown("### 🧠 Model and evaluation")
        st.write(
            "Model yang digunakan pada halaman prediksi adalah LightGBM Regressor, "
            "yang dipilih karena menghasilkan MAE, MSE, RMSE, dan MAPE terendah serta "
            "R² Test tertinggi pada pengujian kelompok."
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("MAE", "$138.71")
        metric_cols[1].metric("RMSE", "$198.33")
        metric_cols[2].metric("MAPE", "7.37%")
        metric_cols[3].metric("R² Test", "0.8809")
        st.markdown("### 📚 Dataset")
        st.write("Computer Prices 2025 · 100,000 records · 33 original columns.")
        st.markdown("### 👥 Team")
        st.write(
            "- Reta Hadiana Unggula — XGBoost Regressor\n"
            "- Saniyatul Warda — LightGBM Regressor"
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;padding:.5rem 0 1rem 0;">
        <div style="font-family:'Space Grotesk';font-weight:700;color:#174A8B;">CompuPrice ✦</div>
        <div class="small-note">Machine Learning Practicum · Group 5 · Universitas Airlangga</div>
    </div>
    """,
    unsafe_allow_html=True,
)
