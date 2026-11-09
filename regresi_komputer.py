"""
TUGAS PRAKTIKUM MACHINE LEARNING - REGRESI
Kelompok 5
Topik    : Prediksi Harga Komputer & Laptop (Computer Prices 2025)
Dataset  : Computer Prices 2025 (Kaggle: https://www.kaggle.com/competitions/computer-prices-2025/data)
Metode   : Extreme Gradient Boosting (XGBoost Regressor) & Light Gradient Boosting Machine (LightGBM Regressor)
Anggota  : 
  - Reta Hadiana Unggula [434241107] (Bagian: XGBoost Regressor)
  - Saniyatul Warda      [434241101] (Bagian: LightGBM Regressor)
"""

import os
import sys
import time

# Konfigurasi direktori cache matplotlib lokal
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from lightgbm import LGBMRegressor

# Set style visualisasi
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_mape(y_true, y_pred):
    """MAPE dalam persen; target bernilai nol tidak disertakan."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan


def make_cv_sample(X_train, y_train, max_samples=10000):
    """Subset reproducible untuk mempercepat 5-Fold Cross Validation."""
    n = min(max_samples, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), size=n, replace=False)
    return X_train.iloc[idx], y_train.iloc[idx]


def load_data(filepath="computer_prices.csv"):
    """Membaca dan memvalidasi dataset dari file CSV"""
    print("=" * 70)
    print("1. MEMUAT DATASET KOMPUTER & LAPTOP")
    print("=" * 70)
    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat. Dimensi data: {df.shape[0]} baris, {df.shape[1]} kolom.")
    print("\nInformasi Tipe Data & Kolom:")
    df.info(show_counts=True)
    print("\n5 Baris Pertama Data:")
    print(df.head())
    print("\nStatistik Deskriptif Fitur Numerik Utama:")
    print(df.describe().round(2).T[['count', 'mean', 'std', 'min', '50%', 'max']])
    print("\nPengecekan Nilai Hilang (Missing Values):")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada nilai hilang (0 Missing Values pada seluruh 33 kolom).")
    return df


def exploratory_data_analysis(df):
    """Melakukan visualisasi eksplorasi data (EDA) komprehensif"""
    print("\n" + "=" * 70)
    print("2. EKSPLORASI DATA & VISUALISASI (EDA)")
    print("=" * 70)
    
    # Plot 1: Distribusi Harga
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=50, color='#1f77b4', edgecolor='black', alpha=0.6)
    plt.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: USD {df['price'].mean():,.2f}")
    plt.axvline(df['price'].median(), color='green', linestyle='-', linewidth=2, label=f"Median: USD {df['price'].median():,.2f}")
    plt.title("Distribusi Harga Komputer & Laptop (Target Variable)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Harga Jual (USD)", fontsize=12)
    plt.ylabel("Frekuensi Unit Komputer", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_distribution_price.png"))
    plt.close()
    print(" - Plot 1: Distribusi harga target disimpan.")

    # Plot 2: Perbandingan Device Type & Brand vs Price
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df, x='device_type', y='price', ax=axes[0], palette='Set2')
    axes[0].set_title("Perbandingan Harga: Desktop vs Laptop", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Tipe Perangkat", fontsize=11)
    axes[0].set_ylabel("Harga (USD)", fontsize=11)

    brand_order = df.groupby('brand')['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='brand', y='price', order=brand_order, ax=axes[1], palette='Spectral')
    axes[1].set_title("Distribusi Harga Berdasarkan Brand Produsen", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Brand", fontsize=11)
    axes[1].set_ylabel("Harga (USD)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_device_and_brand_vs_price.png"))
    plt.close()
    print(" - Plot 2: Device type & brand vs price disimpan.")

    # Plot 3: Pengaruh RAM dan GPU Tier
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ram_filtered = df[df['ram_gb'].isin([8, 16, 32, 64, 128])]
    sns.boxplot(data=ram_filtered, x='ram_gb', y='price', hue='device_type', palette='coolwarm', ax=axes[0])
    axes[0].set_title("Pengaruh Kapasitas RAM (GB) terhadap Harga", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Kapasitas RAM (GB)", fontsize=11)
    axes[0].set_ylabel("Harga (USD)", fontsize=11)

    sns.boxplot(data=df, x='gpu_tier', y='price', palette='viridis', ax=axes[1])
    axes[1].set_title("Pengaruh Tingkatan GPU (GPU Tier) terhadap Harga", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Tingkatan GPU (GPU Tier: 1=Entry ... 6=Flagship)", fontsize=11)
    axes[1].set_ylabel("Harga (USD)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_ram_gpu_vs_price.png"))
    plt.close()
    print(" - Plot 3: RAM dan GPU Tier vs price disimpan.")

    # Plot 4: Matriks Korelasi
    numeric_cols = [
        'price', 'ram_gb', 'vram_gb', 'gpu_tier', 'cpu_tier', 
        'cpu_cores', 'cpu_threads', 'cpu_boost_ghz', 'storage_gb', 
        'refresh_hz', 'weight_kg', 'battery_wh', 'warranty_months'
    ]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                cbar_kws={'label': 'Koefisien Korelasi Pearson'}, linewidths=0.5)
    plt.title("Matriks Korelasi Pearson Fitur Hardware terhadap Harga Komputer", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_matrix.png"))
    plt.close()
    print(" - Plot 4: Heatmap korelasi Pearson disimpan.")


def prepare_data(df):
    """Menyiapkan fitur X, target y, dan preprocessor ColumnTransformer"""
    print("\n" + "=" * 70)
    print("3. PIPELINE PREPROCESSING & TRAIN-TEST SPLIT")
    print("=" * 70)
    
    # Kolom teks berkardinalitas tinggi yang bersifat identifier sintetis dihapus
    # Informasi performa inti CPU & GPU sudah diwakili oleh tier, brand, core, thread, clock, dan vram
    drop_cols = ['model', 'cpu_model', 'price']
    X = df.drop(columns=drop_cols)
    y = df['price']

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Jumlah Fitur Numerik ({len(num_cols)}): {num_cols}")
    print(f"Jumlah Fitur Kategorikal ({len(cat_cols)}): {cat_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Alokasi Train Data : {X_train.shape[0]} baris (80%)")
    print(f"Alokasi Test Data  : {X_test.shape[0]} baris (20%)")
    
    return X, y, X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols


def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, preprocessor):
    """
    Melatih dan mengevaluasi model XGBoost Regressor (Tugas Reta Hadiana Unggula),
    serta membandingkannya dengan baseline Linear Regression.
    """
    print("\n" + "=" * 70)
    print("4. PELATIHAN MODEL & EVALUASI EMPIRIS (XGBOOST REGRESSOR)")
    print("=" * 70)

    # Inisialisasi model
    # Baseline: Linear Regression
    # Main Model: XGBoost Regressor
    models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'XGBoost Regressor (Model Utama)': xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_pipelines = {}

    for name, estimator in models.items():
        print(f"\n---> Memproses Model: {name} ...")
        t0 = time.time()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ])

        # Pelatihan
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Prediksi
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Perhitungan Metrik Evaluasi
        # Perhitungan Metrik Evaluasi
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)

        # Perhitungan MAPE
        y_test_array = np.asarray(y_test)
        mask = y_test_array != 0

        mape = np.mean(
            np.abs(
                (y_test_array[mask] - y_pred_test[mask]) /
                y_test_array[mask]
            )
        ) * 100

        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)

        # 5-Fold Cross Validation R2 pada subset 10,000 data train agar efisien dan representatif
        print(f"     Menghitung 5-Fold Cross Validation...")
        cv_sample_idx = np.random.RandomState(42).choice(len(X_train), size=10000, replace=False)
        X_train_cv = X_train.iloc[cv_sample_idx]
        y_train_cv = y_train.iloc[cv_sample_idx]
        
        cv_scores = cross_val_score(pipeline, X_train_cv, y_train_cv, cv=5, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()

        print(f"     [HASIL {name}]")
        print(f"     - Waktu Pelatihan: {fit_time:.2f} detik")
        print(f"     - MAE       : USD {mae:,.2f}")
        print(f"     - MSE       : {mse:,.2f}")
        print(f"     - RMSE      : USD {rmse:,.2f}")
        print(f"     - MAPE      : {mape:.2f}%")
        print(f"     - R² Test   : {r2_test:.4f}")
        print(f"     - R² Train  : {r2_train:.4f}")
        print(f"     - 5-Fold CV : {cv_mean:.4f} (+/- {cv_scores.std():.4f})")

        results.append({
            'Algoritma': name,
            'MAE (USD)': round(mae, 2),
            'MSE (USD²)': round(mse, 2),
            'RMSE (USD)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'R² Test': round(r2_test, 4),
            'R² Train': round(r2_train, 4),
            '5-Fold CV R²': round(cv_mean, 4)
        })
        trained_pipelines[name] = pipeline

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nRangkuman hasil evaluasi disimpan ke: {results_csv_path}")
    print(results_df.to_string(index=False))

    # Visualisasi Hasil Komparasi Performa
    plot_model_comparison(results_df)

    # Visualisasi Evaluasi Mendalam XGBoost (Actual vs Pred, Residual, Feature Importance)
    xgb_pipeline = trained_pipelines['XGBoost Regressor (Model Utama)']
    plot_xgboost_evaluation(xgb_pipeline, X_test, y_test, preprocessor)

    return trained_pipelines, results_df


def plot_model_comparison(results_df):
    """Grafik perbandingan lima metrik evaluasi."""
    metric_info = [
        ('MAE (USD)', 'MAE (USD) - lebih rendah lebih baik'),
        ('MSE (USD²)', 'MSE (USD²) - lebih rendah lebih baik'),
        ('RMSE (USD)', 'RMSE (USD) - lebih rendah lebih baik'),
        ('MAPE (%)', 'MAPE (%) - lebih rendah lebih baik'),
        ('R² Test', 'R² Test - lebih tinggi lebih baik')
    ]
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    for ax, (col, ylabel) in zip(axes, metric_info):
        sns.barplot(data=results_df, x='Algoritma', y=col, ax=ax)
        ax.set_title(f'Perbandingan {col}', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        for bar in ax.patches:
            val = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        (bar.get_x() + bar.get_width()/2, val),
                        ha='center', va='bottom', fontsize=8,
                        xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_model_comparison.png"), bbox_inches='tight')
    plt.close()
    print(" - Plot komparasi lima metrik model disimpan.")

def plot_xgboost_evaluation(pipeline, X_test, y_test, preprocessor):
    """Visualisasi evaluasi mendalam untuk model XGBoost Regressor"""
    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred

    # 1. Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(9, 7))
    sample_size = min(3000, len(y_test))
    idx = np.random.RandomState(42).choice(len(y_test), size=sample_size, replace=False)
    
    plt.scatter(y_test.iloc[idx], y_pred[idx], alpha=0.35, color='#2b5c8f', edgecolors='none', label='Sampel Uji')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prediksi Ideal (y = ŷ)')
    
    plt.title("XGBoost Regressor: Nilai Aktual vs Nilai Prediksi", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Harga Aktual (USD)", fontsize=12)
    plt.ylabel("Harga Hasil Prediksi (USD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_actual_vs_predicted.png"))
    plt.close()
    print(" - Plot Actual vs Predicted XGBoost disimpan.")

    # 2. Residual Plot & Histogram Distribusi Error
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_pred[idx], residuals.iloc[idx], alpha=0.35, color='#d95f02', edgecolors='none')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[0].set_title("Residual vs Nilai Prediksi", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Harga Prediksi (USD)", fontsize=11)
    axes[0].set_ylabel("Residual (Aktual - Prediksi)", fontsize=11)

    sns.histplot(residuals, kde=True, bins=50, color='#7570b3', ax=axes[1], edgecolor='black', alpha=0.6)
    axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean Error: {residuals.mean():.2f}")
    axes[1].set_title("Distribusi Normalitas Galat (Residual Error)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Nilai Residual Error (USD)", fontsize=11)
    axes[1].set_ylabel("Frekuensi", fontsize=11)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_residuals_plot.png"))
    plt.close()
    print(" - Plot Residual Analysis XGBoost disimpan.")

    # 3. Feature Importance
    try:
        xgb_model = pipeline.named_steps['regressor']
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = xgb_model.feature_importances_
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)

        # Bersihkan nama fitur agar mudah dibaca di grafik
        fi_df['Feature_Clean'] = fi_df['Feature'].apply(lambda x: x.replace('num__', '').replace('cat__', ''))

        plt.figure(figsize=(10, 7))
        sns.barplot(data=fi_df, x='Importance', y='Feature_Clean', palette='viridis')
        plt.title("Top 15 Feature Importance (XGBoost Regressor)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Tingkat Kepentingan Relatif (Gain)", fontsize=12)
        plt.ylabel("Nama Fitur Spesifikasi Hardware", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "eval_feature_importance.png"))
        plt.close()
        print(" - Plot Feature Importance XGBoost disimpan.")
    except Exception as e:
        print("Feature importance extraction warning:", e)



def lightgbm_model_template(X_train, y_train, X_test, y_test, preprocessor):
    """Training, testing, evaluasi, dan visualisasi LightGBM. Bagian Saniyatul Warda."""
    print("\\n" + "=" * 70)
    print("5. PELATIHAN MODEL & EVALUASI LIGHTGBM REGRESSOR")
    print("=" * 70)

    model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    print("\\n[INFO] Memulai training LightGBM...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start

    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)

    print("[INFO] Menghitung 5-Fold Cross Validation...")
    X_cv, y_cv = make_cv_sample(X_train, y_train)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_cv, y_cv, cv=cv, scoring='r2', n_jobs=-1
    )
    cv_mean = cv_scores.mean()

    print("\\nHASIL EVALUASI LIGHTGBM")
    print(f"Waktu Training : {fit_time:.2f} detik")
    print(f"MAE            : USD {mae:,.2f}")
    print(f"MSE            : {mse:,.2f}")
    print(f"RMSE           : USD {rmse:,.2f}")
    print(f"MAPE           : {mape:.2f}%")
    print(f"R² Test        : {r2_test:.4f}")
    print(f"R² Train       : {r2_train:.4f}")
    print(f"5-Fold CV R²   : {cv_mean:.4f} (+/- {cv_scores.std():.4f})")

    # Actual vs predicted dan residual
    residuals = np.asarray(y_test) - pred_test
    sample_n = min(5000, len(y_test))
    idx = np.random.RandomState(42).choice(len(y_test), size=sample_n, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(np.asarray(y_test)[idx], pred_test[idx], alpha=0.35)
    lo = min(np.min(y_test), np.min(pred_test))
    hi = max(np.max(y_test), np.max(pred_test))
    axes[0].plot([lo, hi], [lo, hi], 'r--')
    axes[0].set_title("LightGBM: Aktual vs Prediksi")
    axes[0].set_xlabel("Harga Aktual (USD)")
    axes[0].set_ylabel("Harga Prediksi (USD)")

    axes[1].scatter(pred_test[idx], residuals[idx], alpha=0.35)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title("LightGBM: Residual Plot")
    axes[1].set_xlabel("Harga Prediksi (USD)")
    axes[1].set_ylabel("Residual (USD)")
    plt.tight_layout()
    eval_path = os.path.join(OUTPUT_DIR, "lightgbm_evaluation.png")
    plt.savefig(eval_path, bbox_inches='tight')
    plt.close()

    # Feature importance
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    fitted_model = pipeline.named_steps['regressor']
    feature_names = fitted_preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': fitted_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    importance_df['Feature_Clean'] = (
        importance_df['Feature'].str.replace('num__', '', regex=False)
        .str.replace('cat__', '', regex=False)
    )
    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x='Importance', y='Feature_Clean')
    plt.title("Top 15 Feature Importance - LightGBM")
    plt.xlabel("Feature Importance")
    plt.ylabel("Fitur")
    plt.tight_layout()
    importance_path = os.path.join(OUTPUT_DIR, "lightgbm_feature_importance.png")
    plt.savefig(importance_path, bbox_inches='tight')
    plt.close()

    metrics = {
        'Algoritma': 'LightGBM Regressor',
        'MAE (USD)': round(mae, 2),
        'MSE (USD²)': round(mse, 2),
        'RMSE (USD)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R² Test': round(r2_test, 4),
        'R² Train': round(r2_train, 4),
        '5-Fold CV R²': round(cv_mean, 4)
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "lightgbm_metrics.csv"), index=False
    )
    print(f"[INFO] Grafik evaluasi: {eval_path}")
    print(f"[INFO] Feature importance: {importance_path}")
    return pipeline, metrics

def simulate_new_predictions(pipeline):
    """
    Mensimulasikan prediksi estimasi harga komputer untuk 3 profil spesifikasi baru:
    1. Laptop Mahasiswa Hemat (Budget Student Laptop)
    2. PC Desktop Desain Grafis & Gaming Menengah (Mid-range Creator/Gaming Desktop)
    3. Workstation Flagship Performa Ekstrem (High-End Workstation)
    """
    print("\n" + "=" * 70)
    print("6. SIMULASI PREDIKSI INFERENSI KASUS SPESIFIKASI BARU")
    print("=" * 70)

    sample_cases = [
        {
            'Profil': 'Laptop Mahasiswa Hemat (Budget Student)',
            'device_type': 'Laptop',
            'brand': 'Acer',
            'release_year': 2024,
            'os': 'Windows',
            'form_factor': 'Mainstream',
            'cpu_brand': 'Intel',
            'cpu_tier': 2,
            'cpu_cores': 6,
            'cpu_threads': 12,
            'cpu_base_ghz': 2.0,
            'cpu_boost_ghz': 3.4,
            'gpu_brand': 'Intel',
            'gpu_model': 'Integrated',
            'gpu_tier': 1,
            'vram_gb': 0,
            'ram_gb': 16,
            'storage_type': 'SSD',
            'storage_gb': 512,
            'storage_drive_count': 1,
            'display_type': 'IPS',
            'display_size_in': 15.6,
            'resolution': '1920x1080',
            'refresh_hz': 60,
            'battery_wh': 50.0,
            'charger_watts': 65.0,
            'psu_watts': 0.0,
            'wifi': 'Wi-Fi 6',
            'bluetooth': 5.2,
            'weight_kg': 1.75,
            'warranty_months': 24
        },
        {
            'Profil': 'PC Desktop Creator & Gaming Menengah',
            'device_type': 'Desktop',
            'brand': 'MSI',
            'release_year': 2024,
            'os': 'Windows',
            'form_factor': 'ATX',
            'cpu_brand': 'AMD',
            'cpu_tier': 4,
            'cpu_cores': 12,
            'cpu_threads': 24,
            'cpu_base_ghz': 3.2,
            'cpu_boost_ghz': 4.4,
            'gpu_brand': 'NVIDIA',
            'gpu_model': 'RTX 40 60',
            'gpu_tier': 4,
            'vram_gb': 8,
            'ram_gb': 32,
            'storage_type': 'NVMe',
            'storage_gb': 1024,
            'storage_drive_count': 2,
            'display_type': 'IPS',
            'display_size_in': 27.0,
            'resolution': '2560x1440',
            'refresh_hz': 144,
            'battery_wh': 0.0,
            'charger_watts': 0.0,
            'psu_watts': 650.0,
            'wifi': 'Wi-Fi 6E',
            'bluetooth': 5.3,
            'weight_kg': 9.20,
            'warranty_months': 36
        },
        {
            'Profil': 'Workstation Flagship Performa Ekstrem',
            'device_type': 'Desktop',
            'brand': 'Dell',
            'release_year': 2025,
            'os': 'Linux',
            'form_factor': 'Tower',
            'cpu_brand': 'Intel',
            'cpu_tier': 6,
            'cpu_cores': 24,
            'cpu_threads': 48,
            'cpu_base_ghz': 3.4,
            'cpu_boost_ghz': 4.5,
            'gpu_brand': 'NVIDIA',
            'gpu_model': 'RTX 40 80',
            'gpu_tier': 6,
            'vram_gb': 16,
            'ram_gb': 64,
            'storage_type': 'NVMe',
            'storage_gb': 2048,
            'storage_drive_count': 3,
            'display_type': 'OLED',
            'display_size_in': 32.0,
            'resolution': '3840x2160',
            'refresh_hz': 165,
            'battery_wh': 0.0,
            'charger_watts': 0.0,
            'psu_watts': 1000.0,
            'wifi': 'Wi-Fi 7',
            'bluetooth': 5.3,
            'weight_kg': 14.50,
            'warranty_months': 36
        }
    ]

    sim_df = pd.DataFrame(sample_cases)
    profil_labels = sim_df.pop('Profil')

    preds = pipeline.predict(sim_df)

    sim_result = pd.DataFrame({
        'Profil Kasus': profil_labels,
        'Tipe': sim_df['device_type'],
        'Brand': sim_df['brand'],
        'CPU Cores/Tier': sim_df['cpu_cores'].astype(str) + " cores (Tier " + sim_df['cpu_tier'].astype(str) + ")",
        'RAM': sim_df['ram_gb'].astype(str) + " GB",
        'Storage': sim_df['storage_gb'].astype(str) + " GB " + sim_df['storage_type'],
        'GPU Tier / VRAM': "Tier " + sim_df['gpu_tier'].astype(str) + " (" + sim_df['vram_gb'].astype(str) + "GB)",
        'Estimasi Harga (USD)': [f"${p:,.2f}" for p in preds]
    })

    sim_csv_path = os.path.join(OUTPUT_DIR, "simulasi_prediksi_baru.csv")
    sim_result.to_csv(sim_csv_path, index=False)
    print("\nHasil Simulasi Prediksi Harga Komputer Baru:")
    print(sim_result.to_string(index=False))
    print(f"\nHasil simulasi disimpan ke: {sim_csv_path}")


def main():
    print("=" * 70)
    print("EKSEKUSI PIPELINE PEMODELAN REGRESI - PREDIKSI HARGA KOMPUTER 2025")
    print("Mata Kuliah: Machine Learning Practicum")
    print("Dosen Pengampu: Barry Nuqoba, S.Si., M.Kom., Ph.D.")
    print("=" * 70)

    # 1. Pemuatan Data
    df = load_data("computer_prices.csv")

    # 2. EDA
    exploratory_data_analysis(df)

    # 3. Preprocessing
    X, y, X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = prepare_data(df)

    # 4. Pelatihan & Evaluasi XGBoost dan baseline
    trained_pipelines, xgb_results_df = train_and_evaluate_xgboost(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # 5. Pelatihan & Evaluasi LightGBM
    lgbm_pipeline, lgbm_metrics = lightgbm_model_template(
        X_train, y_train, X_test, y_test, preprocessor
    )
    trained_pipelines['LightGBM Regressor'] = lgbm_pipeline

    # 6. Gabungkan hasil evaluasi seluruh model
    results_df = pd.concat(
        [xgb_results_df, pd.DataFrame([lgbm_metrics])],
        ignore_index=True
    )
    results_csv_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("\\n" + "=" * 70)
    print("PERBANDINGAN AKHIR SELURUH MODEL")
    print("=" * 70)
    print(results_df.to_string(index=False))
    plot_model_comparison(results_df)

    # Gunakan model dengan MAE terkecil untuk simulasi prediksi
    # 6. Gunakan model dengan MAE terkecil
    best_name = results_df.loc[
        results_df['MAE (USD)'].idxmin(), 'Algoritma'
    ]

    print(f"\n[INFO] Model terbaik berdasarkan MAE: {best_name}")

    # Ambil pipeline model terbaik
    best_pipeline = trained_pipelines[best_name]

    # Simpan model terbaik beserta preprocessing
    import joblib

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "best_model.joblib")

    joblib.dump(best_pipeline, model_path)

    # Simpan nama algoritma terbaik
    with open(
        os.path.join(MODEL_DIR, "best_model_name.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(best_name)

    print(f"[INFO] Model berhasil disimpan: {model_path}")

    # Simulasi prediksi menggunakan model terbaik
    simulate_new_predictions(best_pipeline)

    print("\n" + "=" * 70)
    print("SELURUH PROSES PEMODELAN & EVALUASI SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()

