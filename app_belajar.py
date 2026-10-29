import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Mobile Price Classification App", layout="wide")

st.title("📱 Aplikasi Klasifikasi Rentang Harga Ponsel")
st.markdown("Aplikasi ini mengadaptasi alur kerja Machine Learning untuk mengklasifikasikan rentang harga ponsel berdasarkan spesifikasi perangkat keras.")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("1. Unggah Data")
train_file = st.sidebar.file_uploader("Unggah File train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Unggah File test.csv", type=["csv"])

if train_file is not None:
    train_df = pd.read_csv(train_file)
    
    with st.expander("Tinjauan Data Latih", expanded=True):
        st.write(f"**Ukuran data latih:** {train_df.shape}")
        st.dataframe(train_df.head())

    if test_file is not None:
        test_df = pd.read_csv(test_file)
        with st.expander("Tinjauan Data Uji"):
            st.write(f"**Ukuran data uji:** {test_df.shape}")
            st.dataframe(test_df.head())
    else:
        test_df = None

    # --- VISUALISASI DATA ---
    st.header("Visualisasi Data")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Distribusi Target
    sns.countplot(ax=axes[0], x='price_range', data=train_df, hue='price_range', palette='viridis', legend=False)
    axes[0].set_title('Distribusi Target (Price Range) di Train Set')
    axes[0].set_xlabel('Price Range')
    axes[0].set_ylabel('Jumlah')

    # 2. Perbandingan RAM (jika data test tersedia)
    if test_df is not None and 'ram' in test_df.columns:
        train_ram = train_df[['ram']].copy()
        train_ram['dataset'] = 'Train'
        test_ram = test_df[['ram']].copy()
        test_ram['dataset'] = 'Test'
        combined_df = pd.concat([train_ram, test_ram], axis=0)

        sns.kdeplot(ax=axes[1], data=combined_df, x='ram', hue='dataset', fill=True, common_norm=False, palette='crest')
        axes[1].set_title('Perbandingan Distribusi RAM (Train vs Test)')
        axes[1].set_xlabel('RAM')
        axes[1].set_ylabel('Kepadatan (Density)')
    else:
        axes[1].text(0.5, 0.5, "Unggah data uji untuk melihat perbandingan RAM", 
                     horizontalalignment='center', verticalalignment='center')
        axes[1].set_title("Perbandingan Distribusi RAM")

    plt.tight_layout()
    st.pyplot(fig)

    # 3. Distribusi Battery Power
    if test_df is not None and 'battery_power' in test_df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        train_batt = train_df[['battery_power']].copy()
        train_batt['dataset'] = 'Train'
        test_batt = test_df[['battery_power']].copy()
        test_batt['dataset'] = 'Test'
        combined_battery = pd.concat([train_batt, test_batt], axis=0)
        
        sns.boxplot(data=combined_battery, x='dataset', y='battery_power', hue='dataset', palette='Set2')
        ax2.set_title('Perbandingan Distribusi Battery Power (Train vs Test)')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Battery Power')
        st.pyplot(fig2)

    # --- PELATIHAN MODEL ---
    st.header("Pelatihan & Evaluasi Model")
    if st.button("Jalankan Proses Pelatihan Model"):
        with st.spinner("Sedang memproses dan melatih model..."):
            # Persiapan Data
            X = train_df.drop(columns=['price_range'])
            y = train_df['price_range']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Standarisasi
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Inisialisasi Model
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss