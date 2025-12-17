import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import io
import os   



# -------------- BILINGUAL SUPPORT --------------
if "language" not in st.session_state:
    st.session_state.language = "English"

def t(en: str, idn: str) -> str:
    """Simple translation helper."""
    return en if st.session_state.language == "English" else idn


# -------------- CONFIG --------------
st.set_page_config(
    page_title=t("Matrix Transformations & Image Processing",
                 "Transformasi Matriks & Pengolahan Citra"),
    layout="wide",
)



import streamlit as st


# Language selector (sidebar)
st.sidebar.selectbox(
    "🌐 Language / Bahasa",
    ["English", "Indonesia"],
    index=0 if st.session_state.language == "English" else 1,
    key="language",
)


# -------------- TABS --------------
tab1, tab2, tab3 = st.tabs([
    t("🔢 Matrix Transformations", "🔢 Transformasi Matriks"),
    t("🖼️ Image Processing", "🖼️ Pengolahan Citra"),
    t("👥 Developer Team", "👥 Tim Pengembang"),
])


# =====================================================================
# TAB 1 – MATRIX TRANSFORMATIONS (kode asli + sedikit teks bilingual)
# =====================================================================
with tab1:
    st.title("🔢 Matrix Transformations Visualizer")
    st.markdown(
        t(
            "**Interactive 2D Linear Algebra Tool** – visualize "
            r"$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ on a unit square, grid, and vectors.",
            "**Alat Aljabar Linear 2D Interaktif** – memvisualisasikan "
            r"$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ pada persegi satuan, grid, dan vektor.",
        )
    )

    # -------------- SESSION STATE --------------
    if "A" not in st.session_state:
        st.session_state.A = np.eye(2, dtype=float)   # linear part
    if "b" not in st.session_state:
        st.session_state.b = np.array([0.0, 0.0], dtype=float)   # translation

    # -------------- SIDEBAR: MATRIX A --------------
    st.sidebar.header(t("🎛️ Linear Part (Matrix A)",
                        "🎛️ Bagian Linear (Matriks A)"))

    col1, col2 = st.sidebar.columns(2)
    a11 = col1.number_input("a₁₁", value=float(st.session_state.A[0, 0]),
                            step=0.1, format="%.2f")
    a12 = col2.number_input("a₁₂", value=float(st.session_state.A[0, 1]),
                            step=0.1, format="%.2f")
    a21 = col1.number_input("a₂₁", value=float(st.session_state.A[1, 0]),
                            step=0.1, format="%.2f")
    a22 = col2.number_input("a₂₂", value=float(st.session_state.A[1, 1]),
                            step=0.1, format="%.2f")

    A = np.array([[a11, a12],
                  [a21, a22]], dtype=float)
    st.session_state.A = A.copy()

    det_A = float(np.linalg.det(A))
    trace_A = float(np.trace(A))

    st.sidebar.markdown("<div style='text-align:center;'>",
                        unsafe_allow_html=True)
    st.latex(
        rf"A = \begin{{pmatrix}}"
        rf"{A[0,0]:.2f} & {A[0,1]:.2f} \\"
        rf"{A[1,0]:.2f} & {A[1,1]:.2f}"
        r"\end{pmatrix}"
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.metric("🔢 det(A)", f"{det_A:.3f}")
    st.sidebar.metric("📏 trace(A)", f"{trace_A:.3f}")

    # -------------- SIDEBAR: TRANSLATION b --------------
    st.sidebar.header(t("📍 Translation (Vector b)",
                        "📍 Translasi (Vektor b)"))

    tx = st.sidebar.number_input("b₁ (translate x)",
                                 value=float(st.session_state.b[0]),
                                 step=0.1, format="%.2f")
    ty = st.sidebar.number_input("b₂ (translate y)",
                                 value=float(st.session_state.b[1]),
                                 step=0.1, format="%.2f")
    b = np.array([tx, ty], dtype=float)
    st.session_state.b = b.copy()

    st.sidebar.markdown("<div style='text-align:center;'>",
                        unsafe_allow_html=True)
    st.latex(r"A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # -------------- PRESET TRANSFORMATIONS --------------
    st.sidebar.header(t("✨ Preset Transformations",
                        "✨ Transformasi Prasetel"))

    mode = st.sidebar.radio(
        t("Choose group:", "Pilih grup:"),
        [
            t("Basic: Scaling / Rotation / Shearing",
              "Dasar: Skala / Rotasi / Geser"),
            t("Reflections", "Refleksi"),
        ]
    )

    # BASIC: Scaling / Rotation / Shearing
    if mode.startswith("Basic") or mode.startswith("Dasar"):
        st.sidebar.markdown("**Scaling**")
        s_x = st.sidebar.slider("Scale X (sₓ)", 0.1, 3.0, 1.0, 0.1)
        s_y = st.sidebar.slider("Scale Y (sᵧ)", 0.1, 3.0, 1.0, 0.1)

        st.sidebar.markdown("**Rotation**")
        angle_deg = st.sidebar.slider("Angle θ (degrees)",
                                      -180.0, 180.0, 0.0, 1.0)
        theta = np.radians(angle_deg)

        st.sidebar.markdown("**Shearing**")
        sh_x = st.sidebar.slider("Shear X (kₓ)", -2.0, 2.0, 0.0, 0.1)
        sh_y = st.sidebar.slider("Shear Y (kᵧ)", -2.0, 2.0, 0.0, 0.1)

        if st.sidebar.button(
            t("Apply Scale + Rotate + Shear",
              "Terapkan Skala + Rotasi + Geser")
        ):
            S = np.array([[s_x, 0.0],
                          [0.0, s_y]], dtype=float)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]], dtype=float)
            H = np.array([[1.0, sh_x],
                          [sh_y, 1.0]], dtype=float)
            A_basic = H @ R @ S
            st.session_state.A = A_basic
            A = A_basic.copy()

    # REFLECTIONS
    if mode.startswith("Reflections") or mode.startswith("Refleksi"):
        st.sidebar.markdown("**Reflection matrices**")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.sidebar.button("X-axis"):
                st.session_state.A = np.array([[1.0, 0.0],
                                               [0.0, -1.0]], dtype=float)
            if st.sidebar.button("Y-axis"):
                st.session_state.A = np.array([[-1.0, 0.0],
                                               [0.0, 1.0]], dtype=float)
        with c2:
            if st.sidebar.button("Origin"):
                st.session_state.A = np.array([[-1.0, 0.0],
                                               [0.0, -1.0]], dtype=float)
            if st.sidebar.button("Line y = x"):
                st.session_state.A = np.array([[0.0, 1.0],
                                               [1.0, 0.0]], dtype=float)

    # update A setelah preset
    A = st.session_state.A.copy()
    det_A = float(np.linalg.det(A))

    # -------------- TEST VECTOR & ANIMATION --------------
    st.sidebar.header(t("🧪 Test & Animation",
                        "🧪 Uji & Animasi"))

    test_x = st.sidebar.number_input("Test vector x",
                                     value=0.5, step=0.1, format="%.2f")
    test_y = st.sidebar.number_input("Test vector y",
                                     value=0.5, step=0.1, format="%.2f")
    t_anim = st.sidebar.slider(
        "Animation t (0 = I, 1 = A, b)",
        0.0, 1.0, 1.0, 0.01
    )

    A_t = (1.0 - t_anim) * np.eye(2) + t_anim * A      # animasi linear part
    b_t = t_anim * b                                   # animasi translasi

    test_vec = np.array([test_x, test_y], dtype=float)
    test_t = A_t @ test_vec + b_t

    # -------------- GEOMETRY (UNIT SQUARE, GRID, BASIS) --------------
    def make_geometry():
        square = np.array([[0, 0],
                           [1, 0],
                           [1, 1],
                           [0, 1],
                           [0, 0]], dtype=float)
        e1 = np.array([[0, 0],
                       [1, 0]], dtype=float)
        e2 = np.array([[0, 0],
                       [0, 1]], dtype=float)
        xs = np.linspace(-0.5, 1.5, 9)
        ys = np.linspace(-0.5, 1.5, 9)
        X, Y = np.meshgrid(xs, ys)
        grid = np.column_stack([X.ravel(), Y.ravel()])
        return square, e1, e2, grid

    square, e1, e2, grid = make_geometry()

    # T(x) = A_t x + b_t
    square_t = (A_t @ square.T).T + b_t
    e1_t = (A_t @ e1.T).T + b_t
    e2_t = (A_t @ e2.T).T + b_t
    grid_t = (A_t @ grid.T).T + b_t

    # -------------- PLOTS --------------
    colL, colR = st.columns(2)

    with colL:
        st.subheader(t("📐 Original Space",
                       "📐 Ruang Asal"))
        fig1, ax1 = plt.subplots(figsize=(5, 5))

        ax1.plot(square[:, 0], square[:, 1], "k-", linewidth=3,
                 label="Unit square")
        ax1.fill(square[:, 0], square[:, 1],
                 color="lightgray", alpha=0.5)

        ax1.arrow(0, 0, 1, 0, head_width=0.05,
                  color="blue", linewidth=3, label="e₁")
        ax1.arrow(0, 0, 0, 1, head_width=0.05,
                  color="green", linewidth=3, label="e₂")

        ax1.scatter(grid[:, 0], grid[:, 1],
                    color="gray", s=20, alpha=0.6)

        ax1.set_aspect("equal", "box")
        ax1.set_xlim(-0.7, 1.7)
        ax1.set_ylim(-0.7, 1.7)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.axvline(0, color="black", linewidth=0.5)
        ax1.set_title("Original unit square & basis")
        ax1.legend(loc="upper right")
        st.pyplot(fig1)

    with colR:
        st.subheader(
            t("✨ Transformed Space  (T(x) = A x + b)",
              "✨ Ruang Tertransformasi  (T(x) = A x + b)")
        )
        fig2, ax2 = plt.subplots(figsize=(5, 5))

        ax2.plot(square_t[:, 0], square_t[:, 1], "r-", linewidth=3,
                 label="T(unit square)")
        ax2.fill(square_t[:, 0], square_t[:, 1],
                 color="red", alpha=0.25)

        ax2.arrow(b_t[0], b_t[1],
                  e1_t[1, 0] - b_t[0], e1_t[1, 1] - b_t[1],
                  head_width=0.07, color="blue",
                  linewidth=3, label="T(e₁)")
        ax2.arrow(b_t[0], b_t[1],
                  e2_t[1, 0] - b_t[0], e2_t[1, 1] - b_t[1],
                  head_width=0.07, color="green",
                  linewidth=3, label="T(e₂)")

        ax2.scatter(grid_t[:, 0], grid_t[:, 1],
                    color="orange", s=18, alpha=0.8,
                    label="T(grid)")

        ax2.arrow(b_t[0], b_t[1],
                  test_t[0] - b_t[0], test_t[1] - b_t[1],
                  head_width=0.09, color="lime", linewidth=3,
                  label=f"T([{test_x:.1f}, {test_y:.1f}])")

        ax2.set_aspect("equal", "box")
        ax2.set_xlim(-4.0, 4.0)
        ax2.set_ylim(-4.0, 4.0)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_title(f"Transformed (det(A) = {det_A:.3f})")
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

    # -------------- METRICS & FORMULAS --------------
    st.markdown("---")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("📐 Area scaling = |det(A)|", f"{abs(det_A):.3f}")

    with m2:
        st.metric(
            "🎯 Test vector",
            f"({test_x:.2f}, {test_y:.2f})",
            f"T(x) = ({test_t[0]:.2f}, {test_t[1]:.2f})",
        )

    with m3:
        st.latex(r"T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}")
        st.latex(r"\det(A) = ad - bc")
        st.latex(r"\text{Area scaling} = |\det(A)|")

    # -------------- EXPORT CSV --------------
    st.markdown("---")
    if st.sidebar.button("📥 Export grid data as CSV"):
        df = pd.DataFrame({
            "x_original": grid[:, 0],
            "y_original": grid[:, 1],
            "x_transformed": grid_t[:, 0],
            "y_transformed": grid_t[:, 1],
        })
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            "Download CSV",
            data=csv,
            file_name="matrix_transformations.csv",
            mime="text/csv",
        )

    st.markdown(
        t(
            "*✅ Translation, Scaling, Rotation, Shearing, Reflection – ready to run with `streamlit run app.py`*",
            "*✅ Translasi, Skala, Rotasi, Geser, Refleksi – siap dijalankan dengan `streamlit run app.py`*",
        )
    )


# =====================================================================
# TAB 2 – IMAGE PROCESSING (blur, sharpen, background removal)
# =====================================================================
with tab2:
    st.header(t("🖼️ Image Processing",
                "🖼️ Pengolahan Citra"))

    uploaded_file = st.file_uploader(
        t("Upload an image (PNG/JPG/JPEG)",
          "Unggah gambar (PNG/JPG/JPEG)"),
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(t("📥 Original Image",
                           "📥 Gambar Asli"))
            st.image(image, use_column_width=True)

        st.markdown("---")
        st.subheader(t("⚙️ Processing Options",
                       "⚙️ Opsi Pemrosesan"))

        blur_radius = st.slider(
            t("Blur (Gaussian radius)", "Blur (radius Gaussian)"),
            0.0, 10.0, 0.0, 0.5
        )
        sharpen_factor = st.slider(
            t("Sharpen factor", "Faktor penajaman"),
            0.0, 3.0, 1.0, 0.1
        )
        brightness_factor = st.slider(
            t("Brightness", "Kecerahan"),
            0.0, 2.0, 1.0, 0.1
        )
        contrast_factor = st.slider(
            t("Contrast", "Kontras"),
            0.0, 2.0, 1.0, 0.1
        )
        remove_bg = st.checkbox(
            t("🗑️ Remove background (simple, optional)",
              "🗑️ Hapus latar belakang (sederhana, opsional)")
        )

        if st.button(t("✨ Apply Processing",
                       "✨ Terapkan Pemrosesan")):
            processed = image.copy()

            # blur
            if blur_radius > 0:
                processed = processed.filter(
                    ImageFilter.GaussianBlur(radius=blur_radius)
                )

            # sharpen
            if sharpen_factor != 1.0:
                enhancer_sharp = ImageEnhance.Sharpness(processed)
                processed = enhancer_sharp.enhance(sharpen_factor)

            # brightness & contrast
            enhancer_bright = ImageEnhance.Brightness(processed)
            processed = enhancer_bright.enhance(brightness_factor)

            enhancer_contrast = ImageEnhance.Contrast(processed)
            processed = enhancer_contrast.enhance(contrast_factor)

            # simple background removal (white-ish → transparent)
            if remove_bg:
                np_img = np.array(processed)
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                # threshold tinggi → anggap background terang sebagai putih
                _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                np_img = cv2.bitwise_and(np_img, np_img, mask=mask_inv)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2RGBA)
                processed = Image.fromarray(np_img)

            with col2:
                st.subheader(t("✨ Processed Image",
                               "✨ Gambar Hasil"))
                st.image(processed, use_column_width=True)

                # download processed image
                buf = io.BytesIO()
                processed.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    t("💾 Download Processed Image",
                      "💾 Unduh Gambar Hasil"),
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png",
                )
    else:
        st.info(
            t("Please upload an image to start.",
              "Silakan unggah gambar untuk memulai.")
        )


# =====================================================================
# TAB 3 – DEVELOPER TEAM PROFILE (AUTO PHOTO - FIXED)
# =====================================================================
with tab3:
    st.header(t("👥 Developer Team", "👥 Tim Pengembang"))

    team = [
        {
            "name": "Nadilla Novi Anggraini",
            "role": "Leader",
            "student": "004202400083",
            "contribution": "Project manager, geometric transformation module",
            "photo": "assets/nadilla.jpg",
        },
        {
            "name": "Ahmad Arda Syafi",
            "role": "Member",
            "student": "004202400005",
            "contribution": "Histogram module, image processing functions",
            "photo": "assets/arda.jpg",
        },
        {
            "name": "Laurensius Mahendra Wisnu Wardana",
            "role": "Member",
            "student": "004202400017",
            "contribution": "Image filtering module, UI/UX design",
            "photo": "assets/laurensius.jpg",
        },
    ]

    for member in team:
        st.markdown("---")
        col_photo, col_info = st.columns([1, 3])

        with col_photo:
            if os.path.exists(member["photo"]):
                img = Image.open(member["photo"])
                st.image(img, width=220)
            else:
                st.image(
                    "https://via.placeholder.com/220x220?text=No+Photo",
                    width=220
                )

        with col_info:
            st.subheader(member["name"])
            st.write(f"**Role:** {member['role']}")
            st.write(f"**Student ID:** {member['student']}")
            st.write(f"**Contribution:** {member['contribution']}")
