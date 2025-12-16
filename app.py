import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Matrix Transformations",
    page_icon="🖼️",
    layout="wide"
)

# ==========================================
# 1. KAMUS BAHASA (TRANSLATION DICTIONARY)
# ==========================================
translations = {
    "en": {
        "nav_title": "Navigation",
        "nav_options": ["Home", "Image Processing Tools", "Team Members"],
        "lang_label": "Language / Bahasa",
        
        # Home Page
        "home_title": "Matrix Transformations in Image Processing",
        "home_overview_title": "🎓 Project Overview",
        "home_desc": """
        Welcome to our Streamlit Web Application! This project demonstrates the fundamental concepts 
        of Computer Vision: **Geometric Transformations** and **Image Filtering**.
        
        We utilize Linear Algebra (Matrices) to manipulate images pixel-by-pixel.
        """,
        "home_sec1": "1. Geometric Transformations",
        "home_sec1_desc": "Geometric operations map pixel coordinates $(x, y)$ to new coordinates $(x', y')$.",
        "home_sec1_note": "Where $\mathbf{M}$ is a $3 \\times 3$ transformation matrix.",
        "home_sec2": "2. Convolution (Filtering)",
        "home_sec2_desc": "Convolution applies a kernel (a small matrix) to every pixel and its neighbors.",
        "home_sec2_note": "Used for Blurring, Sharpening, and Edge Detection.",
        "home_info": "👈 Navigate to the 'Image Processing Tools' page to try it out!",

        # Tools Page
        "tools_title": "🛠️ Image Processing Tools",
        "upload_label": "Upload an Image",
        "select_op": "Select Transformation",
        "ops_list": ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Blur Filter", "Sharpen Filter"],
        "params": "Parameters",
        "orig_img": "Original Image",
        "trans_img": "Transformed Image",
        "warn_upload": "Please upload an image via the sidebar to begin.",
        "matrix_geo": "Geometric Transformation Matrix ($3 \\times 3$):",
        "matrix_ker": "Convolution Kernel Used:",
        
        # Parameter Labels
        "p_shift_x": "Shift X (pixels)",
        "p_shift_y": "Shift Y (pixels)",
        "p_scale_x": "Scale X",
        "p_scale_y": "Scale Y",
        "p_angle": "Angle (degrees)",
        "p_shear_x": "Shear X",
        "p_shear_y": "Shear Y",
        "p_axis": "Reflection Axis",
        "p_kernel": "Kernel Size (Odd number)",
        "p_strength": "Sharpen Strength",

        # Team Page
        "team_title": "👥 The Team",
        "team_intro": "Meet the developers behind this Matrix Transformation application.",
        "role_label": "Role:",
        "about_title": "💡 How We Built This",
        "about_desc": """
        **Team Contribution Summary:**
        This application is the result of a collaborative effort. We divided the work into **User Interface**, **Matrix Logic**, **Convolution Filters**, and **System Integration**. 
        
        Each member contributed to ensuring that the mathematical matrices displayed correspond accurately to the visual transformations applied to the uploaded images.
        """
    },
    "id": {
        "nav_title": "Navigasi",
        "nav_options": ["Beranda", "Alat Pengolah Citra", "Anggota Tim"],
        "lang_label": "Language / Bahasa",

        # Home Page
        "home_title": "Transformasi Matriks dalam Pengolahan Citra",
        "home_overview_title": "🎓 Ringkasan Proyek",
        "home_desc": """
        Selamat datang di Aplikasi Web Streamlit kami! Proyek ini mendemonstrasikan konsep dasar 
        Computer Vision: **Transformasi Geometris** dan **Pemfilteran Citra**.
        
        Kami menggunakan Aljabar Linear (Matriks) untuk memanipulasi gambar piksel demi piksel.
        """,
        "home_sec1": "1. Transformasi Geometris",
        "home_sec1_desc": "Operasi geometris memetakan koordinat piksel $(x, y)$ ke koordinat baru $(x', y')$.",
        "home_sec1_note": "Di mana $\mathbf{M}$ adalah matriks transformasi $3 \\times 3$.",
        "home_sec2": "2. Konvolusi (Filtering)",
        "home_sec2_desc": "Konvolusi menerapkan kernel (matriks kecil) ke setiap piksel dan tetangganya.",
        "home_sec2_note": "Digunakan untuk Pemburaman (Blur), Penajaman (Sharpen), dan Deteksi Tepi.",
        "home_info": "👈 Buka halaman 'Alat Pengolah Citra' untuk mencobanya!",

        # Tools Page
        "tools_title": "🛠️ Alat Pengolah Citra",
        "upload_label": "Unggah Gambar",
        "select_op": "Pilih Transformasi",
        "ops_list": ["Translasi", "Skala", "Rotasi", "Shearing (Geser)", "Refleksi", "Filter Blur", "Filter Sharpen"],
        "params": "Parameter",
        "orig_img": "Gambar Asli",
        "trans_img": "Gambar Hasil",
        "warn_upload": "Silakan unggah gambar melalui sidebar untuk memulai.",
        "matrix_geo": "Matriks Transformasi Geometris ($3 \\times 3$):",
        "matrix_ker": "Kernel Konvolusi yang Digunakan:",

        # Parameter Labels
        "p_shift_x": "Geser X (piksel)",
        "p_shift_y": "Geser Y (piksel)",
        "p_scale_x": "Skala X",
        "p_scale_y": "Skala Y",
        "p_angle": "Sudut (derajat)",
        "p_shear_x": "Shear X",
        "p_shear_y": "Shear Y",
        "p_axis": "Sumbu Refleksi",
        "p_kernel": "Ukuran Kernel (Ganjil)",
        "p_strength": "Kekuatan Penajaman",

        # Team Page
        "team_title": "👥 Anggota Tim",
        "team_intro": "Kenalan dengan pengembang di balik aplikasi Transformasi Matriks ini.",
        "role_label": "Peran:",
        "about_title": "💡 Bagaimana Kami Membuat Ini",
        "about_desc": """
        **Ringkasan Kontribusi Tim:**
        Aplikasi ini adalah hasil kerja sama tim. Kami membagi tugas menjadi **Antarmuka Pengguna**, **Logika Matriks**, **Filter Konvolusi**, dan **Integrasi Sistem**. 
        
        Setiap anggota berkontribusi untuk memastikan bahwa matriks matematika yang ditampilkan sesuai dengan transformasi visual yang diterapkan pada gambar yang diunggah.
        """
    }
}

# --- Sidebar Language Selector ---
st.sidebar.title("Settings / Pengaturan")
lang_choice = st.sidebar.radio("Language / Bahasa", ["English", "Bahasa Indonesia"])
lang = "en" if lang_choice == "English" else "id"
t = translations[lang] # Shortcut variable

# --- Helper Functions (LOGIC - TIDAK BERUBAH) ---

def load_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)
    return img_array

def get_translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

def get_scaling_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)

def get_rotation_matrix(angle_degrees, center_x, center_y):
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    t_origin = get_translation_matrix(-center_x, -center_y)
    t_back = get_translation_matrix(center_x, center_y)
    return t_back @ rot_matrix @ t_origin

def get_shear_matrix(shx, shy):
    return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float32)

def get_reflection_matrix(axis, width, height):
    if axis == 'x': 
        return np.array([[-1, 0, width], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    elif axis == 'y': 
        return np.array([[1, 0, 0], [0, -1, height], [0, 0, 1]], dtype=np.float32)
    return np.eye(3)

def apply_geometric_transform(image, matrix):
    rows, cols = image.shape[:2]
    transformed = cv2.warpPerspective(image, matrix, (cols, rows))
    return transformed

def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# --- Navigation ---
st.sidebar.markdown("---")
# PERHATIKAN DISINI: Variable 't' digunakan, bukan teks biasa.
st.sidebar.title(t["nav_title"]) 
page = st.sidebar.radio("Go to", t["nav_options"]) 

# ==========================================
# PAGE 1: HOME
# ==========================================
# Cek pakai index (urutan) agar tidak error saat bahasa diganti
if page == t["nav_options"][0]: 
    st.title(t["home_title"])
    st.markdown(f"### {t['home_overview_title']}")
    st.write(t["home_desc"])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header(t["home_sec1"])
        st.write(t["home_sec1_desc"])
        st.latex(r"\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{M} \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}")
        st.write(t["home_sec1_note"])
        
    with col2:
        st.header(t["home_sec2"])
        st.write(t["home_sec2_desc"])
        st.latex(r"g(x, y) = \omega * f(x, y)")
        st.write(t["home_sec2_note"])

    st.info(t["home_info"])

# ==========================================
# PAGE 2: TOOLS
# ==========================================
elif page == t["nav_options"][1]:
    st.title(t["tools_title"])
    
    st.sidebar.markdown("---")
    st.sidebar.header(t["params"])
    
    uploaded_file = st.sidebar.file_uploader(t["upload_label"], type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        rows, cols = original_image.shape[:2]
        
        operation = st.sidebar.selectbox(t["select_op"], t["ops_list"])
        
        st.sidebar.subheader(t["params"])
        
        processed_image = None
        matrix_to_show = None
        
        # --- Logic for each operation ---
        op_index = t["ops_list"].index(operation)

        if op_index == 0: # Translation
            tx = st.sidebar.slider(t["p_shift_x"], -200, 200, 50)
            ty = st.sidebar.slider(t["p_shift_y"], -200, 200, 50)
            matrix_to_show = get_translation_matrix(tx, ty)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif op_index == 1: # Scaling
            sx = st.sidebar.slider(t["p_scale_x"], 0.1, 3.0, 1.0)
            sy = st.sidebar.slider(t["p_scale_y"], 0.1, 3.0, 1.0)
            matrix_to_show = get_scaling_matrix(sx, sy)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif op_index == 2: # Rotation
            angle = st.sidebar.slider(t["p_angle"], -180, 180, 45)
            matrix_to_show = get_rotation_matrix(angle, cols/2, rows/2)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif op_index == 3: # Shearing
            shx = st.sidebar.slider(t["p_shear_x"], -1.0, 1.0, 0.2)
            shy = st.sidebar.slider(t["p_shear_y"], -1.0, 1.0, 0.0)
            matrix_to_show = get_shear_matrix(shx, shy)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif op_index == 4: # Reflection
            axis = st.sidebar.radio(t["p_axis"], ["x", "y"])
            matrix_to_show = get_reflection_matrix(axis, cols, rows)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif op_index == 5: # Blur Filter
            k_size = st.sidebar.slider(t["p_kernel"], 3, 25, 5, step=2)
            kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)
            processed_image = apply_convolution(original_image, kernel)
            matrix_to_show = kernel 
            
        elif op_index == 6: # Sharpen Filter
            strength = st.sidebar.slider(t["p_strength"], 1, 3, 1)
            base = -1 * strength
            center = 4 * strength + 1 
            kernel = np.array([
                [0, base, 0],
                [base, center, base],
                [0, base, 0]
            ], dtype=np.float32)
            processed_image = apply_convolution(original_image, kernel)
            matrix_to_show = kernel

        # --- Display Area ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t["orig_img"])
            st.image(original_image, use_column_width=True)
        with col2:
            st.subheader(t["trans_img"])
            if processed_image is not None:
                st.image(processed_image, use_column_width=True)
            else:
                st.write("Adjust parameters to see the result.")
        
        # Show Matrix/Kernel
        st.markdown("---")
        if matrix_to_show is not None:
            if op_index >= 5: # Filter operations
                st.markdown(f"##### {t['matrix_ker']}")
            else:
                st.markdown(f"##### {t['matrix_geo']}")
            st.write(matrix_to_show)

    else:
        st.warning(t["warn_upload"])

# ==========================================
# PAGE 3: TEAM
# ==========================================
elif page == t["nav_options"][2]:
    st.title(t["team_title"])
    st.write(t["team_intro"])
    
    st.markdown("---")
    def display_member(name, role, description, image_path):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if os.path.exists(image_path):
                    st.image(image_path, width=150)
                else:
                    st.error("Img Not Found")
                    st.caption(image_path)
            
            with col2:
                st.subheader(name)
                st.markdown(f"**{t['role_label']}** {role}")
                st.write(description)
            
            st.markdown("---")

    # 1. Hamzah
    display_member(
        "Hamzah Sholehudin Yusuf",
        "Lead Developer & Architecture",
        "Responsible for the overall application architecture, project coordination, and implementing core matrix transformation logic.",
        "hamzah.jpg"
    )

    # 2. Affan
    display_member(
        "Muhammad Affan Rasyidin",
        "Frontend & UI/UX Specialist",
        "Designed the user interface using Streamlit, ensuring a responsive layout and intuitive sidebar navigation for image processing tools.",
        "affan.jpg"
    )

    # 3. Darrel
    display_member(
        "Muhammad Darrel Yashaq",
        "Algorithm Engineer (Geometric)",
        "Focused on implementing the geometric transformation algorithms (Rotation, Scaling, Shearing) and matrix math verification.",
        "darrel.jpg"
    )

    # 4. Emil
    display_member(
        "Muhammad Emil Lutfi",
        "Algorithm Engineer (Filtering)",
        "Developed the convolution kernel logic for Blurring and Sharpening filters and conducted testing/debugging of the application.",
        "emil.jpg"
    )

    # Penutup
    st.header(t["about_title"])
    st.info(t["about_desc"])
