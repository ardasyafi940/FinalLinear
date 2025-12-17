import streamlit as st

# Judul aplikasi
st.title("Kalkulator Sederhana")

# Input angka
angka1 = st.number_input("Masukkan angka pertama:", value=0.0)
angka2 = st.number_input("Masukkan angka kedua:", value=0.0)

# Pilihan operasi
operasi = st.selectbox(
    "Pilih operasi:",
    ("Tambah (+)", "Kurang (-)", "Kali (×)", "Bagi (÷)")
)

# Tombol hitung
if st.button("Hitung"):
    if operasi == "Tambah (+)":
        hasil = angka1 + angka2
    elif operasi == "Kurang (-)":
        hasil = angka1 - angka2
    elif operasi == "Kali (×)":
        hasil = angka1 * angka2
    elif operasi == "Bagi (÷)":
        if angka2 != 0:
            hasil = angka1 / angka2
        else:
            hasil = "Error: Pembagian dengan nol!"

    # Tampilkan hasil
    st.success(f"Hasil: {hasil}")
