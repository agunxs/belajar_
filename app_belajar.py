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

st.set_page_config(page_title="Mobile Price Classification", layout="wide")

st.title("📱 Aplikasi Klasifikasi Rentang Harga Ponsel")
st.markdown("Aplikasi interaktif untuk mengeksplorasi data, melatih model Machine Learning, dan melakukan prediksi harga ponsel.")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("1. Unggah Data")
train_file = st.sidebar.file_uploader("Unggah File train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Unggah File test.csv (Opsional)", type=["csv"])

if train_file is not None:
    train_df = pd.read_csv(train_file)
    
    # Membuat Tab agar tampilan UI lebih rapi
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data", "⚙️ Pelatihan Model", "📂 Prediksi File", "🎛️ Prediksi Interaktif"])
    
    with tab1:
        st.header("Visualisasi & Eksplorasi Data")
        with st.expander("Lihat Data Latih", expanded=True):
            st.write(f"**Ukuran data latih:** {train_df.shape}")
            st.dataframe(train_df.head())

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.countplot(ax=axes[0], x='price_range', data=train_df, hue='price_range', palette='viridis', legend=False)
        axes[0].set_title('Distribusi Target (Price Range)')

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            train_ram = train_df[['ram']].copy()
            train_ram['dataset'] = 'Train'
            test_ram = test_df[['ram']].copy()
            test_ram['dataset'] = 'Test'
            combined_df = pd.concat([train_ram, test_ram], axis=0)

            sns.kdeplot(ax=axes[1], data=combined_df, x='ram', hue='dataset', fill=True, palette='crest')
            axes[1].set_title('Distribusi RAM (Train vs Test)')
        else:
            sns.histplot(ax=axes[1], data=train_df, x='ram', kde=True, color='blue')
            axes[1].set_title('Distribusi RAM (Train)')

        st.pyplot(fig)

    with tab2:
        st.header("Pelatihan & Evaluasi Model")
        if st.button("Jalankan Pelatihan Model"):
            with st.spinner("Sedang memproses dan melatih model..."):
                X = train_df.drop(columns=['price_range'])
                y = train_df['price_range']
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

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

                    results[name] = {
                        'model_object': model,
                        'accuracy': accuracy_score(y_val, y_pred),
                        'report': classification_report(y_val, y_pred, output_dict=True)
                    }

                df_results = pd.DataFrame({
                    'Model': results.keys(),
                    'Accuracy': [info['accuracy'] for info in results.values()]
                }).sort_values(by='Accuracy', ascending=False)
                
                st.subheader("Peringkat Akurasi Model")
                st.dataframe(df_results.style.highlight_max(subset=['Accuracy'], color='lightgreen'))

                best_model_name = df_results.iloc[0]['Model']
                st.session_state['best_model'] = results[best_model_name]['model_object']
                st.session_state['scaler'] = scaler
                st.session_state['features'] = X.columns.tolist()

                st.success(f"Model terbaik yang disimpan: **{best_model_name}**")

    with tab3:
        st.header("Prediksi Batch (Data Uji)")
        if test_file is not None:
            if 'best_model' in st.session_state:
                test_df = pd.read_csv(test_file)
                X_test = test_df.drop(columns=['id'], errors='ignore')
                X_test_scaled = st.session_state['scaler'].transform(X_test)
                
                predictions = st.session_state['best_model'].predict(X_test_scaled)
                test_df['predicted_price'] = predictions
                
                st.write("### Sebagian Hasil Prediksi")
                st.dataframe(test_df[['id', 'battery_power', 'ram', 'predicted_price']].head(10) if 'id' in test_df.columns else test_df.head())
            else:
                st.warning("Silakan latih model terlebih dahulu di tab 'Pelatihan Model'.")
        else:
            st.info("Unggah file test.csv di sidebar untuk melakukan prediksi batch.")

    with tab4:
        st.header("🎛️ Prediksi Interaktif (Ubah Variabel)")
        if 'best_model' in st.session_state:
            st.write("Ubah nilai spesifikasi di bawah ini untuk melihat prediksi harga ponsel secara langsung.")
            
            # Membuat input dinamis untuk setiap fitur
            user_input = {}
            cols = st.columns(4)
            features = st.session_state['features']
            
            for i, col in enumerate(features):
                with cols[i % 4]:
                    min_val = float(train_df[col].min())
                    max_val = float(train_df[col].max())
                    mean_val = float(train_df[col].median())
                    
                    # Jika data biner (0 dan 1), gunakan selectbox
                    if train_df[col].nunique() == 2:
                        user_input[col] = st.selectbox(f"{col}", options=[0, 1], index=0)
                    else:
                        user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            
            st.markdown("---")
            if st.button("🔮 Prediksi Sekarang", use_container_width=True):
                # Konversi input ke DataFrame
                input_df = pd.DataFrame([user_input])
                input_scaled = st.session_state['scaler'].transform(input_df)
                
                # Prediksi
                prediction = st.session_state['best_model'].predict(input_scaled)[0]
                
                st.success(f"### 📱 Hasil Prediksi Rentang Harga (Price Range): **{prediction}**")
        else:
            st.warning("Silakan latih model terlebih dahulu di tab 'Pelatihan Model' untuk membuka fitur ini.")
else:
    st.info("👋 Selamat datang! Silakan unggah file `train.csv` di panel sebelah kiri untuk memulai.")
