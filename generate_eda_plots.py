import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Memuat dataset computer_prices.csv...")
df = pd.read_csv("computer_prices.csv")
print(f"Dimensi data: {df.shape[0]} baris, {df.shape[1]} kolom.")

# 1. Distribusi Target (Price)
print("Membuat plot 1: Distribusi Harga Komputer...")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=50, color='#1f77b4', ax=ax, edgecolor='black', alpha=0.6)
ax.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${df['price'].mean():,.2f}")
ax.axvline(df['price'].median(), color='green', linestyle='-', linewidth=2, label=f"Median: ${df['price'].median():,.2f}")
ax.set_title("Distribusi Harga Komputer & Laptop (Target Variable)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Harga Jual (USD $)", fontsize=12)
ax.set_ylabel("Frekuensi Unit Komputer", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distribution_price.png"))
plt.close()

# 2. Perbandingan Harga Berdasarkan Tipe Perangkat & Brand
print("Membuat plot 2: Harga Berdasarkan Brand & Tipe Perangkat...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Boxplot Device Type vs Price
sns.boxplot(data=df, x='device_type', y='price', palette='Set2', ax=axes[0])
axes[0].set_title("Perbandingan Harga: Desktop vs Laptop", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Tipe Perangkat", fontsize=11)
axes[0].set_ylabel("Harga (USD $)", fontsize=11)

# Boxplot Brand vs Price (sorted by median price)
brand_order = df.groupby('brand')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='brand', y='price', order=brand_order, palette='Spectral', ax=axes[1])
axes[1].set_title("Distribusi Harga Berdasarkan Brand Produsen", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Brand", fontsize=11)
axes[1].set_ylabel("Harga (USD $)", fontsize=11)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_device_and_brand_vs_price.png"))
plt.close()

# 3. Hubungan RAM & GPU Tier terhadap Harga
print("Membuat plot 3: Pengaruh RAM dan GPU Tier...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Lineplot / Pointplot RAM vs Price
ram_filtered = df[df['ram_gb'].isin([8, 16, 32, 64, 128])]
sns.boxplot(data=ram_filtered, x='ram_gb', y='price', hue='device_type', palette='coolwarm', ax=axes[0])
axes[0].set_title("Pengaruh Kapasitas RAM (GB) terhadap Harga", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Kapasitas RAM (GB)", fontsize=11)
axes[0].set_ylabel("Harga (USD $)", fontsize=11)

# Boxplot GPU Tier vs Price
sns.boxplot(data=df, x='gpu_tier', y='price', palette='viridis', ax=axes[1])
axes[1].set_title("Pengaruh Tingkatan GPU (GPU Tier) terhadap Harga", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Tingkatan GPU (GPU Tier: 1=Entry ... 6=Flagship)", fontsize=11)
axes[1].set_ylabel("Harga (USD $)", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_ram_gpu_vs_price.png"))
plt.close()

# 4. Heatmap Matriks Korelasi Pearson Fitur Kuantitatif
print("Membuat plot 4: Heatmap Matriks Korelasi...")
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

# 5. Distribusi Spesifikasi Komponen (CPU Brand, Storage Type, OS, Display)
print("Membuat plot 5: Komposisi Fitur Kategorikal Utama...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x='cpu_brand', palette='pastel', ax=axes[0, 0])
axes[0, 0].set_title("Proporsi Brand Prosesor (CPU Brand)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Brand CPU", fontsize=10)
axes[0, 0].set_ylabel("Jumlah Unit", fontsize=10)

sns.countplot(data=df, x='storage_type', palette='pastel', ax=axes[0, 1])
axes[0, 1].set_title("Proporsi Tipe Penyimpanan (Storage Type)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Tipe Media Penyimpanan", fontsize=10)
axes[0, 1].set_ylabel("Jumlah Unit", fontsize=10)

sns.countplot(data=df, x='os', palette='pastel', ax=axes[1, 0])
axes[1, 0].set_title("Proporsi Sistem Operasi (Operating System)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Sistem Operasi", fontsize=10)
axes[1, 0].set_ylabel("Jumlah Unit", fontsize=10)

sns.countplot(data=df, x='display_type', palette='pastel', ax=axes[1, 1])
axes[1, 1].set_title("Proporsi Panel Layar (Display Type)", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Tipe Panel Layar", fontsize=10)
axes[1, 1].set_ylabel("Jumlah Unit", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_specs_breakdown.png"))
plt.close()

print("Seluruh visualisasi EDA berhasil disimpan ke folder:", OUTPUT_DIR)

