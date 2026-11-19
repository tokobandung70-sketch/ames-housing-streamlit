from pathlib import Path
import math
import pickle

import pandas as pd
import streamlit as st
from sklearn.base import is_classifier

# File model .pkl harus berada dalam folder yang sama dengan app.py.
FOLDER_APLIKASI = Path(__file__).resolve().parent

DAFTAR_MODEL = {
    "Linear Regression": "Linear_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "Random Forest": "Random_Forest.pkl",
}

# 1 square foot = 0.09290304 meter persegi.
M2_PER_SQFT = 0.09290304

# Silakan ganti identitas.
NAMA = "Nama"
NIM = "NIM"


def format_rupiah(nilai: float) -> str:
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


st.set_page_config(
    page_title="Ames Housing Prediction Example",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.3rem; font-weight: 750; margin-bottom: .2rem;}
    .subtitle {color: #687789; margin-bottom: 1.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def muat_model(nama_file: str, waktu_modifikasi: int, ukuran_file: int):
    path = FOLDER_APLIKASI / nama_file
    try:
        with path.open("rb") as file:
            model = pickle.load(file)

        # Menangani file pickle yang menyimpan model di dalam dict.
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        if not hasattr(model, "predict"):
            return None, "Isi .pkl bukan model yang mempunyai metode predict()."

        n_fitur = getattr(model, "n_features_in_", None)
        if n_fitur is not None and int(n_fitur) != 1:
            return None, (
                f"Model membutuhkan {n_fitur} fitur, sedangkan aplikasi hanya mengirim "
                "'Gr Liv Area'. Gunakan model yang dilatih dengan satu fitur."
            )

        nama_fitur = getattr(model, "feature_names_in_", None)
        if nama_fitur is not None and list(nama_fitur) != ["Gr Liv Area"]:
            return None, (
                f"Model mengharapkan fitur {list(nama_fitur)}, "
                "sedangkan aplikasi mengirim ['Gr Liv Area']."
            )

        if is_classifier(model):
            return None, "File berisi model klasifikasi. Harga rumah membutuhkan model regresi."

        return model, None
    except Exception as exc:
        return None, f"Gagal memuat model: {exc}"


def ambil_model(nama_file: str):
    path = FOLDER_APLIKASI / nama_file
    if not path.is_file():
        return None, f"File tidak ditemukan: {path}"
    try:
        data_file = path.stat()
        return muat_model(nama_file, data_file.st_mtime_ns, data_file.st_size)
    except OSError as exc:
        return None, f"Tidak bisa membaca file model: {exc}"


# SIDEBAR
with st.sidebar:
    st.subheader("📁 Status Model")
    for nama, nama_file in DAFTAR_MODEL.items():
        if (FOLDER_APLIKASI / nama_file).is_file():
            st.success(f"{nama}: file tersedia", icon="✅")
        else:
            st.warning(f"{nama}: file belum ditemukan", icon="⚠️")

    st.divider()
    st.subheader("💱 Kurs USD → IDR")
    kurs_idr = st.number_input(
        "Nilai 1 USD dalam rupiah (Rp)",
        min_value=1.0,
        value=16000.0,
        step=100.0,
        format="%.2f",
        help="Kurs diisi manual. Nilai awal Rp 16.000 hanyalah contoh, bukan kurs real-time.",
    )
    st.caption("Kurs diisi manual, bukan kurs real-time. Ubah sesuai kurs yang ingin digunakan.")



# HALAMAN UTAMA
st.markdown('<div class="main-title">🏠 Prediksi Harga Rumah</div>', unsafe_allow_html=True)
st.subheader("Masukkan Luas Bangunan")

kolom_input, kolom_konversi = st.columns(2, gap="large")
with kolom_input:
    luas_sqft = st.number_input(
        "Luas area tempat tinggal (Square feet / ft²)",
        min_value=1.0,
        value=1600.0,
        step=10.0,
        format="%.2f",
        help="Masukkan luas area tempat tinggal di atas tanah (Gr Liv Area) dalam ft².",
    )
with kolom_konversi:
    luas_m2 = luas_sqft * M2_PER_SQFT
    st.metric("Hasil konversi ke meter persegi (m²)", f"{luas_m2:,.2f} m²")

st.caption(f"Nilai yang masuk ke ketiga model: **{luas_sqft:,.2f} ft²** (Gr Liv Area).")
tombol_prediksi = st.button("🔍 Prediksi Harga", type="primary", use_container_width=True)

if tombol_prediksi:
    input_model = pd.DataFrame({"Gr Liv Area": [luas_sqft]})
    hasil = []
    masalah = []

    for nama_model, nama_file in DAFTAR_MODEL.items():
        model, pesan_error = ambil_model(nama_file)
        if pesan_error:
            masalah.append(f"**{nama_model}:** {pesan_error}")
            continue

        try:
            harga_usd = float(model.predict(input_model)[0])
            if not math.isfinite(harga_usd):
                raise ValueError("Prediksi harga bukan angka yang valid.")
            hasil.append({
                "Model": nama_model,
                "Harga USD": harga_usd,
                "Harga IDR": harga_usd * kurs_idr,
            })
        except Exception as exc:
            masalah.append(f"**{nama_model}:** Prediksi gagal: {exc}")

    if masalah:
        st.warning("Sebagian model belum dapat dijalankan:")
        for pesan in masalah:
            st.markdown(f"- {pesan}")

    if hasil:
        st.divider()
        st.subheader("Hasil Prediksi Harga Rumah")
        kolom_hasil = st.columns(len(hasil))
        for kolom, item in zip(kolom_hasil, hasil):
            with kolom:
                with st.container(border=True):
                    st.markdown(f"**{item['Model']}**")
                    st.metric("Harga dalam USD", f"US$ {item['Harga USD']:,.2f}")
                    st.metric("Harga dalam Rupiah", format_rupiah(item["Harga IDR"]))

        st.caption(
            f"Konversi menggunakan kurs yang Anda tentukan: 1 USD = {format_rupiah(kurs_idr)}. "
            "Nilai rupiah hanya konversi mata uang dari prediksi harga USD, bukan "
            "estimasi harga pasar rumah di Indonesia. Dataset model berasal dari Ames Housing."
        )
    else:
        st.error("Belum ada model yang berhasil memprediksi. Periksa file .pkl di folder Streamlit.")

st.divider()
st.caption(f"© {NAMA} — {NIM}")
