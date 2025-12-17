import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64

st.markdown(
    """
    <style>
    .main {
        background-color: #FFD1DC;  /* pink muda */
    }
    div.stButton>button {
        background-color: #FF6B9B;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton>button:hover {
        background-color: #ff4f8a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ===== MULTI LANGUAGE: ID, EN, ZH =====
lang = st.sidebar.selectbox("Language / 语言 / Bahasa", ["ID", "EN", "ZH"])

TEXT = {
    "ID": {
        "title": "2D Transformation Matrix",
        "subtitle": "Upload dataset, lihat statistik, korelasi, dan ekspor laporan PDF.",
        "upload": "Upload file CSV",
        "preview": "Tampilan Awal Dataset",
        "stats": "Statistik Deskriptif",
        "scatter": "Scatter Plot",
        "x_axis": "Sumbu X",
        "y_axis": "Sumbu Y",
        "corr_matrix": "Matriks Korelasi",
        "corr_heatmap": "Heatmap Korelasi",
        "pdf_section": "Ekspor Laporan PDF",
        "pdf_button": "Buat dan download laporan PDF",
        "info_upload": "Silakan upload file CSV untuk mulai analisis.",
        "members_title": "Anggota Kelompok"
    },
    "EN": {
        "title": "2D Transformation Matrix",
        "subtitle": "Upload a dataset, view statistics, correlations, and export a PDF report.",
        "upload": "Upload CSV file",
        "preview": "Dataset Preview",
        "stats": "Descriptive Statistics",
        "scatter": "Scatter Plot",
        "x_axis": "X axis",
        "y_axis": "Y axis",
        "corr_matrix": "Correlation Matrix",
        "corr_heatmap": "Correlation Heatmap",
        "pdf_section": "Export PDF Report",
        "pdf_button": "Generate and download PDF report",
        "info_upload": "Please upload a CSV file to start the analysis.",
        "members_title": "Group Members"
    },
    "ZH": {
        "title": "粉色数据探索应用",
        "subtitle": "上传数据集，查看统计、相关系数，并导出 PDF 报告。",
        "upload": "上传 CSV 文件",
        "preview": "数据集预览",
        "stats": "描述性统计",
        "scatter": "散点图",
        "x_axis": "横轴",
        "y_axis": "纵轴",
        "corr_matrix": "相关系数矩阵",
        "corr_heatmap": "相关系数热力图",
        "pdf_section": "导出 PDF 报告",
        "pdf_button": "生成并下载 PDF 报告",
        "info_upload": "请先上传 CSV 文件以开始分析。",
        "members_title": "小组成员"
    }
}

# ===== SIDEBAR: GROUP MEMBERS =====
st.sidebar.title(TEXT[lang]["members_title"])
st.sidebar.write("- Nadilla Novi Anggraini (Leader)")
st.sidebar.write("- Ahmad Arda Syafi ")
st.sidebar.write("- Laurensius Mahendra Wisnu Wardana ")

# ===== MAIN TITLE =====
st.title(TEXT[lang]["title"])
st.write(TEXT[lang]["subtitle"])

# ===== FILE UPLOADER =====
uploaded_file = st.file_uploader(TEXT[lang]["upload"], type=["csv"])

# ===== PDF HELPER =====
def create_pdf_report(df, corr):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Explorer Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Descriptive statistics (numerical):", ln=True)
    pdf.set_font("Arial", "", 9)

    desc = df.select_dtypes(include=np.number).describe()
    for col in desc.columns:
        pdf.ln(4)
        mean_val = desc[col].get("mean", 0)
        std_val = desc[col].get("std", 0)
        pdf.cell(0, 6, f"{col}: mean={mean_val:.3f}, std={std_val:.3f}", ln=True)

    if corr is not None:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Correlation:", ln=True)
        pdf.set_font("Arial", "", 9)
        for col in corr.columns:
            line = ", ".join(
                [f"{col}-{idx}={corr.loc[idx, col]:.2f}" for idx in corr.index]
            )
            pdf.ln(4)
            pdf.multi_cell(0, 5, line)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

# ===== MAIN LOGIC =====
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader(TEXT[lang]["preview"])
    st.dataframe(df.head())

    # Descriptive statistics
    st.subheader(TEXT[lang]["stats"])
    st.write(df.describe(include="all"))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Scatter plot
    if len(numeric_cols) >= 2:
        st.subheader(TEXT[lang]["scatter"])
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox(TEXT[lang]["x_axis"], numeric_cols, key="xcol")
        with col2:
            y_col = st.selectbox(TEXT[lang]["y_axis"], numeric_cols, key="ycol")

        fig_scatter, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#FF6B9B")
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig_scatter)

    # Correlation
    if len(numeric_cols) >= 2:
        st.subheader(TEXT[lang]["corr_matrix"])
        corr = df[numeric_cols].corr()
        st.write(corr)

        st.subheader(TEXT[lang]["corr_heatmap"])
        fig_corr, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)
    else:
        corr = None

    # PDF export
    st.subheader(TEXT[lang]["pdf_section"])
    if st.button(TEXT[lang]["pdf_button"]):
        pdf_bytes = create_pdf_report(df, corr)
        b64 = base64.b64encode(pdf_bytes).decode("latin-1")
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="data_report.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info(TEXT[lang]["info_upload"])