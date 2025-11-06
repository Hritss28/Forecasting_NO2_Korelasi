import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model_and_features(model_type):
    if model_type == "1_hari":
        model = joblib.load('saved_models/knn_model_day2_filtered.pkl')
        scaler = joblib.load('saved_models/scaler_day2_filtered.pkl')
        sample_data = pd.read_csv('no2_results/day2_supervised_filtered.csv')
        features = [col for col in sample_data.columns if col.startswith('t-')]
    else:  # 3_hari
        model = joblib.load('saved_models/knn_model_day1_filtered.pkl')
        scaler = joblib.load('saved_models/scaler_day1_filtered.pkl')
        sample_data = pd.read_csv('no2_results/day1_supervised_filtered.csv')
        features = [col for col in sample_data.columns if col.startswith('t-')]
    return model, scaler, features

@st.cache_data
def load_thresholds():
    return joblib.load('saved_models/thresholds_day3.pkl')

st.set_page_config(page_title="Prediksi NO₂ Bangkalan", page_icon="🌫️")
st.title("🌫️ Prediksi Kadar NO₂ - Bangkalan")
st.caption("Prediksi kadar NO₂ troposfer (mol/m²) menggunakan model KNN berdasarkan data Sentinel-5P.")

# Pilihan model prediksi
prediksi_type = st.radio(
    "Pilih Jenis Prediksi:",
    ("1_hari", "3_hari"),
    format_func=lambda x: "Prediksi 1 Hari Ke Depan" if x == "1_hari" else "Prediksi 3 Hari Ke Depan",
    horizontal=True
)

model, scaler, features = load_model_and_features(prediksi_type)
thresholds = load_thresholds()

def get_kategori(nilai):
    """Kategorisasi berdasarkan threshold"""
    if nilai <= thresholds['low']:
        return "🟢 **RENDAH**", "Kadar NO₂ rendah, kualitas udara baik."
    elif nilai <= thresholds['medium']:
        return "🟡 **SEDANG**", "Kadar NO₂ sedang, masih dalam batas aman."
    else:
        return "🔴 **TINGGI**", "Kadar NO₂ tinggi, waspadai kualitas udara!"

# Tampilkan features yang digunakan
st.info(f"📊 **Model {prediksi_type.title()}**: Menggunakan features: {', '.join(features)}")

# Input
input_values = {}
cols = st.columns(len(features))

for i, feature in enumerate(features):
    with cols[i]:
        input_values[feature] = st.number_input(
            f"NO₂ Hari Ini ({feature})", 
            value=0.000030, 
            format="%.6f",
            help=f"Data NO₂ pada waktu {feature}"
        )

if st.button("Prediksi", type="primary"):
    X = pd.DataFrame([input_values])[features]
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)[0]
    
    st.success("### Hasil Prediksi")
    
    if isinstance(y_pred, (list, tuple)) or (hasattr(y_pred, 'shape') and len(y_pred.shape) > 0 and y_pred.shape[0] > 1):
        # Multi-output
        num_predictions = len(y_pred)
        cols = st.columns(num_predictions)
        
        for i in range(num_predictions):
            with cols[i]:
                nilai_molm2 = y_pred[i]
                kategori, keterangan = get_kategori(nilai_molm2)
                
                st.metric(f"NO₂ Hari ke {i+1} (t+{i+1})", f"{nilai_molm2:.6f} mol/m²")
                st.markdown(kategori)
                st.caption(keterangan)
    else:
        # Single output
        nilai_molm2 = float(y_pred)
        kategori, keterangan = get_kategori(nilai_molm2)
        
        st.metric("NO₂ Hari Besok (t+1)", f"{nilai_molm2:.6f} mol/m²")
        st.markdown(kategori)
        st.caption(keterangan)

with st.expander("📋 Informasi Model & Threshold"):
    st.markdown(f"""
    **Features yang Digunakan**: {', '.join(features)}
    
    **Alasan Pemilihan**: Berdasarkan analisis korelasi ≥ 50% dengan target
    """)
    
    st.markdown(f"""    
    **Kategori Kadar NO₂:**
    - 🟢 **RENDAH**: ≤ {thresholds['low']:.6f} mol/m²
    - 🟡 **SEDANG**: {thresholds['low']:.6f} - {thresholds['medium']:.6f} mol/m²
    - 🔴 **TINGGI**: > {thresholds['medium']:.6f} mol/m²
    
    **Statistik Data:**
    - Minimum: {thresholds['min']:.6f} mol/m²
    - Maksimum: {thresholds['max']:.6f} mol/m²
    - Rata-rata: {thresholds['mean']:.6f} mol/m²
    """)

st.divider()
st.caption("Model: KNN Regression berdasarkan analisis korelasi | Data: 268 observasi | Metode Threshold: Quartile")