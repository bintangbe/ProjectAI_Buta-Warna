import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from utils import simulate_colorblindness
from fpdf import FPDF
import base64
from io import BytesIO
from transfer_predict import predict_digit_transfer

st.set_page_config(page_title="Deteksi Buta Warna", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #1f2c3e;
        color: white;
    }
    .main, .block-container {
        background-color: #1f2c3e !important;
        color: white !important;
        padding: 2rem 1rem 1rem 1rem;
        border-radius: 10px;
    }
    .stApp {
        background-color: #1f2c3e;
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("🎨 Deteksi Buta Warna - Tes Ishihara (Transfer Learning + Input Manual)")
st.markdown("Upload gambar tes Ishihara, lalu sistem akan memprediksi angka secara otomatis dan Anda juga dapat memasukkan angka yang Anda lihat untuk dibandingkan.")

# Upload gambar
with st.container():
    st.subheader("📤 Upload Gambar Tes Ishihara")
    uploaded_file = st.file_uploader("Pilih file gambar (.jpg/.png)", type=["jpg", "png"])

hasil_diagnosa = ""
input_user = ""
if uploaded_file:
    with st.expander("🖼️ Pratinjau Gambar", expanded=True):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Gambar yang Diunggah', use_column_width=True)

    # Ambil label dari nama file
    filename = uploaded_file.name
    ground_truth = filename.split("_")[0] if "_" in filename else None

    # Prediksi Transfer Learning
    img_array = np.array(img)
    predicted_digit = predict_digit_transfer(img_array)
    st.subheader("🤖 Prediksi Otomatis AI (Transfer Learning)")
    st.write(f"📌 Model Memprediksi Angka: **{predicted_digit}**")

    # Input Manual
    st.subheader("✍️ Masukkan Angka yang Anda Lihat (Manual)")
    input_user = st.text_input("Contoh: 3", max_chars=3)

    # Diagnosa berdasarkan input manual
    if ground_truth and input_user:
        st.subheader("🧠 Diagnosa Berdasarkan Input Manual")
        if input_user.strip() == ground_truth.strip():
            hasil_diagnosa = "Tidak terindikasi buta warna. Input manual sesuai dengan label gambar."
            st.success("✅ " + hasil_diagnosa)
        else:
            hasil_diagnosa = "Kemungkinan mengalami buta warna. Input manual berbeda dari label seharusnya."
            st.error("⚠️ " + hasil_diagnosa)

    st.subheader("👁️ Simulasi Penglihatan untuk Jenis Buta Warna")
    cb_type = st.selectbox("Pilih jenis buta warna yang ingin disimulasikan:", ["protanopia", "deuteranopia", "tritanopia"])
    img_resized = cv2.resize(img_array, (100, 100))
    simulated_img = simulate_colorblindness(img_resized, cb_type)
    st.image(simulated_img, caption=f"Simulasi - {cb_type.capitalize()}", use_column_width=True)

    # Tombol ekspor ke PDF
    if hasil_diagnosa:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.set_title("Hasil Diagnosa Tes Buta Warna")
        pdf.set_creator("Aplikasi AI Ishihara")

        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(0, 10, "Laporan Hasil Tes Buta Warna", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Arial", size=12)
        pdf.cell(50, 10, "Nama File", ln=0)
        pdf.cell(0, 10, f": {filename}", ln=1)

        pdf.cell(50, 10, "Label Seharusnya", ln=0)
        pdf.cell(0, 10, f": {ground_truth}", ln=1)

        if input_user:
            pdf.cell(50, 10, "Input Pengguna", ln=0)
            pdf.cell(0, 10, f": {input_user}", ln=1)

        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Hasil Diagnosa:\n{hasil_diagnosa}")

        buffer = BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="hasil_diagnosa.pdf">📄 Unduh Hasil Diagnosa (PDF)</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small>🛠️ Dibuat untuk Proyek AI - Ishihara Blind Test Cards</small>", unsafe_allow_html=True)


