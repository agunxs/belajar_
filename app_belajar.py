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
                'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'LDA': LDA()
            }

            results = {}
            for name, model in models.items():
                if name in ['Logistic Regression', 'LDA']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                acc = accuracy_score(y_val, y_pred)
                results[name] = {
                    'model_object': model,
                    'accuracy': acc,
                    'report': classification_report(y_val, y_pred, output_dict=True)
                }

            # Tampilkan Hasil Akurasi
            df_results = pd.DataFrame({
                'Model': results.keys(),
                'Accuracy': [info['accuracy'] for info in results.values()]
            }).sort_values(by='Accuracy', ascending=False)
            
            st.subheader("Peringkat Akurasi Model")
            st.dataframe(df_results.style.highlight_max(subset=['Accuracy'], color='lightgreen'))

            # Simpan model terbaik ke session state
            best_model_name = df_results.iloc[0]['Model']
            st.session_state['best_model'] = results[best_model_name]['model_object']
            st.session_state['best_model_name'] = best_model_name
            st.session_state['scaler'] = scaler

            # Evaluasi Mendalam Model Terbaik (Asumsi Regresi Logistik terbaik sesuai notebook)
            eval_model = results['Logistic Regression']['model_object']
            y_val_pred_lr = eval_model.predict(X_val_scaled)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classification Report (Logistic Regression)")
                report_df = pd.DataFrame(results['Logistic Regression']['report']).transpose()
                st.dataframe(report_df)

            with col2:
                st.subheader("Confusion Matrix (Logistic Regression)")
                cm = confusion_matrix(y_val, y_val_pred_lr)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
                disp.plot(cmap='Blues', ax=ax_cm)
                plt.grid(False)
                st.pyplot(fig_cm)

    # --- PREDIKSI DATA UJI ---
    if test_df is not None and 'best_model' in st.session_state:
        st.header("Hasil Prediksi pada Data Pengujian")
        
        # Penyelarasan kolom
        X_test_data = test_df.drop(columns=['id'], errors='ignore')
        
        # Standarisasi data test menggunakan scaler yang telah dilatih
        scaler_loaded = st.session_state['scaler']
        X_test_scaled = scaler_loaded.transform(X_test_data)
        
        # Prediksi menggunakan model terbaik (Regresi Logistik)
        best_model = st.session_state['best_model']
        test_predictions = best_model.predict(X_test_scaled)
        
        test_df_result = test_df.copy()
        test_df_result['predicted_price_range'] = test_predictions
        
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.write("### Sampel Data dengan Prediksi")
            display_cols = ['id', 'battery_power', 'ram', 'predicted_price_range']
            display_cols = [c for c in display_cols if c in test_df_result.columns]
            st.dataframe(test_df_result[display_cols].head(15))
            
        with col_res2:
            st.write("### Distribusi Prediksi")
            st.dataframe(test_df_result['predicted_price_range'].value_counts().sort_index().rename_axis("Price Range").reset_index(name="Jumlah"))

        # Fitur unduh CSV
        csv_data = test_df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=csv_data,
            file_name='hasil_prediksi_ponsel.csv',
            mime='text/csv'
        )
else:
    st.info("Silakan unggah file `train.csv` di panel sisi kiri untuk memulai analisis.")
