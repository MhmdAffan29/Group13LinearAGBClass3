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

# --- Helper Functions ---

def load_image(image_file):
    """Loads an image file and converts it to an OpenCV format (numpy array)."""
    img = Image.open(image_file)
    img_array = np.array(img)
    # Convert RGB to BGR for OpenCV handling if needed, but we stick to RGB for display consistency
    return img_array

def get_translation_matrix(tx, ty):
    """Returns a 3x3 Translation Matrix."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

def get_scaling_matrix(sx, sy):
    """Returns a 3x3 Scaling Matrix."""
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def get_rotation_matrix(angle_degrees, center_x, center_y):
    """
    Returns a 3x3 Rotation Matrix.
    Combines Translation(to origin) -> Rotation -> Translation(back).
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation around origin
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Adjust for center of rotation: T_back * R * T_origin
    t_origin = get_translation_matrix(-center_x, -center_y)
    t_back = get_translation_matrix(center_x, center_y)
    
    return t_back @ rot_matrix @ t_origin

def get_shear_matrix(shx, shy):
    """Returns a 3x3 Shear Matrix."""
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def get_reflection_matrix(axis, width, height):
    """Returns a 3x3 Reflection Matrix."""
    if axis == 'x': # Flip horizontally
        return np.array([
            [-1, 0, width],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    elif axis == 'y': # Flip vertically
        return np.array([
            [1, 0, 0],
            [0, -1, height],
            [0, 0, 1]
        ], dtype=np.float32)
    return np.eye(3)

def apply_geometric_transform(image, matrix):
    """Applies a geometric transform using OpenCV warpPerspective."""
    rows, cols = image.shape[:2]
    transformed = cv2.warpPerspective(image, matrix, (cols, rows))
    return transformed

def apply_convolution(image, kernel):
    """Applies a convolution kernel."""
    return cv2.filter2D(image, -1, kernel)

# --- Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Image Processing Tools", "Team Members"])

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("Matrix Transformations in Image Processing")
    st.markdown("### 🎓 Project Overview")
    st.write("""
    Welcome to our Streamlit Web Application! This project demonstrates the fundamental concepts 
    of Computer Vision: **Geometric Transformations** and **Image Filtering**.
    
    We utilize Linear Algebra (Matrices) to manipulate images pixel-by-pixel.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("1. Geometric Transformations")
        st.write("Geometric operations map pixel coordinates $(x, y)$ to new coordinates $(x', y')$.")
        st.latex(r"\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{M} \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}")
        st.write("Where $\mathbf{M}$ is a $3 \times 3$ transformation matrix.")
        
    with col2:
        st.header("2. Convolution (Filtering)")
        st.write("Convolution applies a kernel (a small matrix) to every pixel and its neighbors.")
        st.latex(r"g(x, y) = \omega * f(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} \omega(s, t) f(x-s, y-t)")
        st.write("Used for Blurring, Sharpening, and Edge Detection.")

    st.info("👈 Navigate to the 'Image Processing Tools' page to try it out!")

# ==========================================
# PAGE 2: TOOLS
# ==========================================
elif page == "Image Processing Tools":
    st.title("🛠️ Image Processing Tools")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        rows, cols = original_image.shape[:2]
        
        operation = st.sidebar.selectbox(
            "Select Transformation",
            ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Blur Filter", "Sharpen Filter"]
        )
        
        st.sidebar.subheader("Parameters")
        
        processed_image = None
        matrix_to_show = None
        
        # --- Logic for each operation ---
        if operation == "Translation":
            tx = st.sidebar.slider("Shift X (pixels)", -200, 200, 50)
            ty = st.sidebar.slider("Shift Y (pixels)", -200, 200, 50)
            matrix_to_show = get_translation_matrix(tx, ty)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif operation == "Scaling":
            sx = st.sidebar.slider("Scale X", 0.1, 3.0, 1.0)
            sy = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.0)
            matrix_to_show = get_scaling_matrix(sx, sy)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif operation == "Rotation":
            angle = st.sidebar.slider("Angle (degrees)", -180, 180, 45)
            matrix_to_show = get_rotation_matrix(angle, cols/2, rows/2)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif operation == "Shearing":
            shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.2)
            shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0)
            matrix_to_show = get_shear_matrix(shx, shy)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif operation == "Reflection":
            axis = st.sidebar.radio("Reflection Axis", ["x", "y"])
            matrix_to_show = get_reflection_matrix(axis, cols, rows)
            processed_image = apply_geometric_transform(original_image, matrix_to_show)
            
        elif operation == "Blur Filter":
            k_size = st.sidebar.slider("Kernel Size (Odd number)", 3, 25, 5, step=2)
            kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)
            processed_image = apply_convolution(original_image, kernel)
            matrix_to_show = kernel 
            
        elif operation == "Sharpen Filter":
            strength = st.sidebar.slider("Sharpen Strength", 1, 3, 1)
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
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        with col2:
            st.subheader("Transformed Image")
            if processed_image is not None:
                st.image(processed_image, use_column_width=True)
            else:
                st.write("Adjust parameters to see the result.")
        
        # Show Matrix/Kernel
        st.markdown("---")
        if matrix_to_show is not None:
            if "Filter" in operation:
                st.markdown("##### Convolution Kernel Used:")
            else:
                st.markdown("##### Geometric Transformation Matrix ($3 \\times 3$):")
            st.write(matrix_to_show)

    else:
        st.warning("Please upload an image via the sidebar to begin.")

# ==========================================
# PAGE 3: TEAM
# ==========================================
elif page == "Team Members":
    st.title("👥 The Team")
    st.write("Meet the developers behind this Matrix Transformation application.")
    
    st.markdown("---")
    def display_member(name, role, description, image_path):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Cek apakah file foto ada
                if os.path.exists(image_path):
                    st.image(image_path, width=150)
                else:
                    # Jika foto tidak ketemu, tampilkan pesan error merah
                    st.error("Foto hilang!")
                    st.caption(f"Cek: {image_path}")
            
            with col2:
                st.subheader(name)
                st.markdown(f"**Role:** {role}")
                st.write(description)
            
            st.markdown("---")
    # 1. Hamzah
    display_member(
        "Hamzah Sholehudin Yusuf",
        "Lead Developer & Architecture",
        "Responsible for the overall application architecture, project coordination, and implementing core matrix transformation logic.",
        "hamzah.jpg"  # <-- Pastikan file 'hamzah.jpg' ada di sebelah file app.py
    )

    # 2. Affan
    display_member(
        "Muhammad Affan Rasyidin",
        "Frontend & UI/UX Specialist",
        "Designed the user interface using Streamlit, ensuring a responsive layout and intuitive sidebar navigation for image processing tools.",
        "affan.jpg"   # <-- Pastikan file 'affan.jpg' ada
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
    st.header("💡 How We Built This")
    st.info("This application is the result of a collaborative effort...")