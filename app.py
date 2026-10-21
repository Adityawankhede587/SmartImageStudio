import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps, ImageFont
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.filters
import skimage.restoration
import rembg


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Smart Image Processing Studio Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #f0f2f6;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .palette-box {
        display: inline-block;
        width: 48px;
        height: 48px;
        border-radius: 8px;
        margin-right: 8px;
        border: 2px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hex_to_rgb(hex_code: str):
    """Convert hex string (e.g. #FFFFFF) to RGB tuple."""
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def extract_color_palette(image_rgb: np.ndarray, n_colors: int = 5):
    """Extract dominant colors using K-Means clustering."""
    pixels = np.float32(image_rgb.reshape(-1, 3))
    # Subsample if image is too large for fast clustering
    if len(pixels) > 50000:
        indices = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.2)
    _, _, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS
    )
    palette = [tuple(map(int, color)) for color in centers]
    return palette


def plot_rgb_histogram(image_rgb: np.ndarray):
    """Generate Matplotlib figure with Red, Green, and Blue histograms."""
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ('#ff4b4b', '#21c354', '#1f77b4')
    labels = ('Red Channel', 'Green Channel', 'Blue Channel')

    for i, (col, lbl) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, label=lbl, linewidth=1.5, alpha=0.85)

    ax.set_xlim([0, 256])
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.legend(facecolor='#1e2229', edgecolor='#444', labelcolor='white', fontsize=8)
    ax.set_title("Color Channel Distribution (RGB)", color='white', fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def apply_vignette(image_np: np.ndarray, strength: float = 0.75):
    """Apply radial vignette dark border to image array."""
    rows, cols = image_np.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / (strength * 2.5))
    kernel_y = cv2.getGaussianKernel(rows, rows / (strength * 2.5))
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignetted = (image_np * mask[:, :, np.newaxis]).astype(np.uint8)
    return vignetted


def apply_sepia(image_np: np.ndarray):
    """Apply classic sepia vintage matrix transform."""
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = cv2.transform(image_np, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


# ============================================================
# APP TITLE & HEADER
# ============================================================

st.markdown('<div class="main-title">🖼️ Smart Image Processing Studio Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced Computer Vision, AI Background Removal, Scientific Enhancements & Multi-step Studio</div>',
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOAD & SESSION STATE SETUP
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, JPEG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_sig = f"{uploaded_file.name}_{len(file_bytes)}"

    # Reset or initialize canvas session state on new file upload
    if "current_file_sig" not in st.session_state or st.session_state.current_file_sig != file_sig:
        original_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.session_state.original_image = original_pil
        st.session_state.canvas = original_pil.copy()
        st.session_state.history = []
        st.session_state.current_file_sig = file_sig

    original_image = st.session_state.original_image
    working_canvas = st.session_state.canvas

    # --------------------------------------------------------
    # TOP TOOLBAR: EDIT PIPELINE & HISTORY
    # --------------------------------------------------------
    t_col1, t_col2, t_col3, t_col4 = st.columns([1.5, 1.2, 1.2, 3])

    with t_col1:
        commit_btn = st.button("💾 Apply Edit to Canvas", help="Permanently chain the current previewed edit into your working canvas")
    with t_col2:
        undo_disabled = len(st.session_state.history) == 0
        undo_btn = st.button("↩️ Undo", disabled=undo_disabled, help="Revert the last committed edit")
    with t_col3:
        reset_btn = st.button("🔄 Reset to Original", help="Discard all edits and revert back to uploaded image")
    with t_col4:
        st.markdown(
            f"**Canvas Size:** `{working_canvas.width} × {working_canvas.height} px` &nbsp;|&nbsp; "
            f"**Edits Chained:** `{len(st.session_state.history)}`"
        )

    # Undo action
    if undo_btn and not undo_disabled:
        st.session_state.canvas = st.session_state.history.pop()
        st.rerun()

    # Reset action
    if reset_btn:
        st.session_state.canvas = st.session_state.original_image.copy()
        st.session_state.history = []
        st.rerun()

    # ========================================================
    # SIDEBAR: CATEGORIES & OPERATIONS
    # ========================================================

    st.sidebar.header("⚙️ Studio Toolkit")

    category = st.sidebar.selectbox(
        "Feature Category",
        [
            "View / Baseline",
            "🤖 AI Background Removal (rembg)",
            "🔬 Scientific & Enhancement (scikit-image)",
            "🎨 Artistic & Aesthetic Filters",
            "⚙️ Basic Lighting & Tone",
            "📐 Geometry & Transform",
            "👁️ Computer Vision & Detection (OpenCV)",
            "✍️ Watermarking & Typography",
            "📊 Analytics & Color Histogram"
        ]
    )

    # By default, preview is based on the active canvas
    preview_image = working_canvas.copy()
    preview_array = np.array(preview_image)
    detection_msg = None

    # ========================================================
    # 1. VIEW / BASELINE
    # ========================================================
    if category == "View / Baseline":
        st.sidebar.info("Viewing active canvas without additional operations. Select a tool from the categories above.")
        preview_image = working_canvas.copy()

    # ========================================================
    # 2. AI BACKGROUND REMOVAL (rembg)
    # ========================================================
    elif category == "🤖 AI Background Removal (rembg)":
        st.sidebar.subheader("AI Background Removal")
        bg_mode = st.sidebar.radio(
            "Background Mode",
            ["Transparent (PNG)", "Solid White", "Solid Black", "Custom Color"]
        )

        custom_color = "#FFFFFF"
        if bg_mode == "Custom Color":
            custom_color = st.sidebar.color_picker("Pick Background Color", "#3498db")

        alpha_matting = st.sidebar.checkbox("Use Fine Alpha Matting", value=True)

        with st.spinner("Removing background with AI (rembg)..."):
            no_bg = rembg.remove(
                working_canvas,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10
            )

            if bg_mode == "Transparent (PNG)":
                preview_image = no_bg
            else:
                color_map = {
                    "Solid White": (255, 255, 255),
                    "Solid Black": (0, 0, 0),
                    "Custom Color": hex_to_rgb(custom_color)
                }
                bg_fill = Image.new("RGBA", no_bg.size, color_map[bg_mode] + (255,))
                bg_fill.paste(no_bg, mask=no_bg.split()[3])
                preview_image = bg_fill.convert("RGB")

    # ========================================================
    # 3. SCIENTIFIC & ENHANCEMENT (scikit-image)
    # ========================================================
    elif category == "🔬 Scientific & Enhancement (scikit-image)":
        sci_op = st.sidebar.selectbox(
            "Scientific Operation",
            [
                "CLAHE (Adaptive Contrast Equalization)",
                "Total Variation Denoising",
                "Otsu Document Binarization (Scanner)",
                "Local Adaptive Thresholding"
            ]
        )

        if sci_op == "CLAHE (Adaptive Contrast Equalization)":
            st.sidebar.caption("Contrast Limited Adaptive Histogram Equalization boosts fine local details without noise blowout.")
            clip_limit = st.sidebar.slider("Clip Limit", 0.01, 0.10, 0.03, step=0.01)
            enhanced = skimage.exposure.equalize_adapthist(preview_array, clip_limit=clip_limit)
            preview_image = Image.fromarray((enhanced * 255).astype(np.uint8))

        elif sci_op == "Total Variation Denoising":
            st.sidebar.caption("Reduces digital noise while preserving crisp boundary edges.")
            weight = st.sidebar.slider("Denoise Weight", 0.02, 0.40, 0.10, step=0.02)
            denoised = skimage.restoration.denoise_tv_chambolle(
                preview_array.astype(float) / 255.0,
                weight=weight,
                channel_axis=-1
            )
            preview_image = Image.fromarray((denoised * 255).astype(np.uint8))

        elif sci_op == "Otsu Document Binarization (Scanner)":
            st.sidebar.caption("Automatically calculates the optimal global threshold for crisp black & white document scanning.")
            gray = cv2.cvtColor(preview_array, cv2.COLOR_RGB2GRAY)
            th_val = skimage.filters.threshold_otsu(gray)
            binary = ((gray > th_val) * 255).astype(np.uint8)
            preview_image = Image.fromarray(binary).convert("RGB")
            st.sidebar.write(f"Computed Otsu Threshold: `{th_val:.1f}`")

        elif sci_op == "Local Adaptive Thresholding":
            st.sidebar.caption("Adapts threshold across uneven illumination.")
            gray = cv2.cvtColor(preview_array, cv2.COLOR_RGB2GRAY)
            block_size = st.sidebar.slider("Block Size (odd)", 11, 101, 35, step=2)
            th_local = skimage.filters.threshold_local(gray, block_size, offset=10)
            binary_local = ((gray > th_local) * 255).astype(np.uint8)
            preview_image = Image.fromarray(binary_local).convert("RGB")

    # ========================================================
    # 4. ARTISTIC & AESTHETIC FILTERS
    # ========================================================
    elif category == "🎨 Artistic & Aesthetic Filters":
        art_op = st.sidebar.selectbox(
            "Artistic Filter",
            [
                "Pencil Sketch",
                "Color Pencil Sketch",
                "Cartoonizer",
                "Sepia Vintage Tone",
                "Vignette Radial Shading",
                "Invert (Negative)"
            ]
        )

        bgr = cv2.cvtColor(preview_array, cv2.COLOR_RGB2BGR)

        if art_op == "Pencil Sketch":
            gray_sketch, _ = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            preview_image = Image.fromarray(gray_sketch).convert("RGB")

        elif art_op == "Color Pencil Sketch":
            _, color_sketch = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            rgb_sketch = cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(rgb_sketch)

        elif art_op == "Cartoonizer":
            # Bilateral filter for color smoothing + adaptive edge mask
            gray = cv2.cvtColor(preview_array, cv2.COLOR_RGB2GRAY)
            gray_blur = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(
                gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
            )
            color = cv2.bilateralFilter(preview_array, d=9, sigmaColor=250, sigmaSpace=250)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            cartoon = cv2.bitwise_and(color, edges_rgb)
            preview_image = Image.fromarray(cartoon)

        elif art_op == "Sepia Vintage Tone":
            preview_image = Image.fromarray(apply_sepia(preview_array))

        elif art_op == "Vignette Radial Shading":
            strength = st.sidebar.slider("Vignette Intensity", 0.3, 1.5, 0.8, step=0.05)
            preview_image = Image.fromarray(apply_vignette(preview_array, strength=strength))

        elif art_op == "Invert (Negative)":
            preview_image = ImageOps.invert(preview_image)

    # ========================================================
    # 5. BASIC LIGHTING & TONE
    # ========================================================
    elif category == "⚙️ Basic Lighting & Tone":
        basic_op = st.sidebar.selectbox(
            "Adjustment",
            ["Brightness", "Contrast", "Color Saturation", "Blur", "Sharpen", "Grayscale"]
        )

        if basic_op == "Brightness":
            val = st.sidebar.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
            enhancer = ImageEnhance.Brightness(working_canvas)
            preview_image = enhancer.enhance(val)

        elif basic_op == "Contrast":
            val = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
            enhancer = ImageEnhance.Contrast(working_canvas)
            preview_image = enhancer.enhance(val)

        elif basic_op == "Color Saturation":
            val = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
            enhancer = ImageEnhance.Color(working_canvas)
            preview_image = enhancer.enhance(val)

        elif basic_op == "Blur":
            amount = st.sidebar.slider("Gaussian Blur Radius", 1, 30, 5)
            preview_image = working_canvas.filter(ImageFilter.GaussianBlur(amount))

        elif basic_op == "Sharpen":
            preview_image = working_canvas.filter(ImageFilter.SHARPEN)

        elif basic_op == "Grayscale":
            preview_image = working_canvas.convert("L").convert("RGB")

    # ========================================================
    # 6. GEOMETRY & TRANSFORM
    # ========================================================
    elif category == "📐 Geometry & Transform":
        geom_op = st.sidebar.selectbox("Transform Operation", ["Resize", "Rotate", "Flip"])

        if geom_op == "Resize":
            maintain_aspect = st.sidebar.checkbox("Maintain Aspect Ratio", value=True)
            orig_w, orig_h = working_canvas.width, working_canvas.height
            aspect_ratio = orig_w / orig_h

            if maintain_aspect:
                new_w = st.sidebar.number_input("Width", min_value=30, max_value=4000, value=orig_w, step=20)
                new_h = int(new_w / aspect_ratio)
                st.sidebar.write(f"Calculated Height: `{new_h} px`")
            else:
                new_w = st.sidebar.number_input("Width", min_value=30, max_value=4000, value=orig_w, step=20)
                new_h = st.sidebar.number_input("Height", min_value=30, max_value=4000, value=orig_h, step=20)

            preview_image = working_canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

        elif geom_op == "Rotate":
            angle = st.sidebar.slider("Angle (°)", -180, 180, 0, step=1)
            expand = st.sidebar.checkbox("Expand Canvas to Fit", value=True)
            preview_image = working_canvas.rotate(angle, expand=expand, resample=Image.Resampling.BICUBIC)

        elif geom_op == "Flip":
            flip_mode = st.sidebar.radio("Direction", ["Horizontal (Left-Right)", "Vertical (Top-Bottom)"])
            if flip_mode.startswith("Horizontal"):
                preview_image = working_canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                preview_image = working_canvas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # ========================================================
    # 7. COMPUTER VISION & DETECTION (OpenCV)
    # ========================================================
    elif category == "👁️ Computer Vision & Detection (OpenCV)":
        cv_op = st.sidebar.selectbox("Vision Tool", ["Face & Eye Detection", "Canny Edge Detection"])

        if cv_op == "Face & Eye Detection":
            box_hex = st.sidebar.color_picker("Face Bounding Box Color", "#00FF00")
            box_color = hex_to_rgb(box_hex)
            # OpenCV uses BGR
            bgr_color = (box_color[2], box_color[1], box_color[0])
            detect_eyes = st.sidebar.checkbox("Also Detect Eyes", value=True)

            bgr = cv2.cvtColor(preview_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            total_eyes = 0
            if detect_eyes:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

            for (x, y, w, h) in faces:
                cv2.rectangle(bgr, (x, y), (x + w, y + h), bgr_color, 3)
                cv2.putText(bgr, "Face", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)

                if detect_eyes:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = bgr[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
                    total_eyes += len(eyes)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 120, 0), 2)

            res_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(res_rgb)
            detection_msg = f"👤 Faces detected: **{len(faces)}**" + (f" | 👁️ Eyes detected: **{total_eyes}**" if detect_eyes else "")

        elif cv_op == "Canny Edge Detection":
            min_val = st.sidebar.slider("Min Threshold", 10, 300, 100)
            max_val = st.sidebar.slider("Max Threshold", 50, 400, 200)
            gray = cv2.cvtColor(preview_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, min_val, max_val)
            preview_image = Image.fromarray(edges).convert("RGB")

    # ========================================================
    # 8. WATERMARKING & TYPOGRAPHY
    # ========================================================
    elif category == "✍️ Watermarking & Typography":
        st.sidebar.subheader("Watermark Settings")
        text = st.sidebar.text_input("Watermark Text", value="Smart Image Studio Pro")
        font_size = st.sidebar.slider("Font Size", 12, 120, 36)
        text_color_hex = st.sidebar.color_picker("Text Color", "#FF0000")
        text_color = hex_to_rgb(text_color_hex)

        pos_mode = st.sidebar.selectbox("Position", ["Bottom-Right", "Bottom-Left", "Center", "Top-Left", "Custom (X, Y)"])

        img_draw = working_canvas.copy()
        draw = ImageDraw.Draw(img_draw)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

        # Calculate bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if pos_mode == "Bottom-Right":
            x_pos = max(10, img_draw.width - text_w - 20)
            y_pos = max(10, img_draw.height - text_h - 20)
        elif pos_mode == "Bottom-Left":
            x_pos = 20
            y_pos = max(10, img_draw.height - text_h - 20)
        elif pos_mode == "Center":
            x_pos = max(0, (img_draw.width - text_w) // 2)
            y_pos = max(0, (img_draw.height - text_h) // 2)
        elif pos_mode == "Top-Left":
            x_pos, y_pos = 20, 20
        else:
            x_pos = st.sidebar.number_input("X Coordinate", 0, max(0, img_draw.width - 1), 20)
            y_pos = st.sidebar.number_input("Y Coordinate", 0, max(0, img_draw.height - 1), 20)

        draw.text((x_pos, y_pos), text, fill=text_color, font=font)
        preview_image = img_draw

    # ========================================================
    # 9. ANALYTICS & COLOR HISTOGRAM (matplotlib)
    # ========================================================
    elif category == "📊 Analytics & Color Histogram":
        st.sidebar.info("Real-time tonal analytics and palette clustering.")
        preview_image = working_canvas.copy()

    # --------------------------------------------------------
    # APPLY / COMMIT EDITS TO PIPELINE
    # --------------------------------------------------------
    if commit_btn:
        st.session_state.history.append(st.session_state.canvas.copy())
        st.session_state.canvas = preview_image.copy()
        st.success("✅ Current edit applied to working canvas! You can continue stacking more edits.")
        st.rerun()

    # ========================================================
    # MAIN DISPLAY: SIDE-BY-SIDE BEFORE / AFTER
    # ========================================================
    if detection_msg:
        st.success(detection_msg)

    st.write("")
    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown("### 📌 Baseline Image")
        st.image(original_image, use_container_width=True)
        st.caption(f"Original Resolution: {original_image.width} × {original_image.height} px")

    with col_proc:
        st.markdown("### ✨ Processed Canvas / Preview")
        st.image(preview_image, use_container_width=True)
        st.caption(f"Current Resolution: {preview_image.width} × {preview_image.height} px")

    # ========================================================
    # ANALYTICS DISPLAY SECTION (Histogram & Palette)
    # ========================================================
    if category == "📊 Analytics & Color Histogram":
        st.markdown("---")
        st.subheader("📊 Color & Channel Analytics")

        a_col1, a_col2 = st.columns(2)

        with a_col1:
            st.markdown("#### RGB Channel Tonal Distribution")
            hist_fig = plot_rgb_histogram(preview_array)
            st.pyplot(hist_fig)

        with a_col2:
            st.markdown("#### Dominant Color Palette (K-Means)")
            palette = extract_color_palette(preview_array, n_colors=5)

            p_cols = st.columns(len(palette))
            for idx, color_rgb in enumerate(palette):
                hex_str = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                with p_cols[idx]:
                    st.markdown(
                        f'<div style="background-color: {hex_str}; height: 60px; border-radius: 8px; border: 2px solid #555;"></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**`{hex_str.upper()}`**")
                    st.caption(f"RGB{color_rgb}")

    # ========================================================
    # EXPORT / DOWNLOAD CENTER
    # ========================================================
    st.markdown("---")
    st.subheader("⬇️ Download Export")

    d_col1, d_col2, d_col3 = st.columns([1.5, 1.5, 3])

    with d_col1:
        export_format = st.selectbox("Export Format", ["PNG", "JPEG", "WEBP"])
    with d_col2:
        quality = 95
        if export_format in ["JPEG", "WEBP"]:
            quality = st.slider("Quality", 20, 100, 90)

    export_bytes = io.BytesIO()
    save_img = preview_image

    if export_format == "PNG":
        save_img.save(export_bytes, format="PNG")
        mime = "image/png"
        ext = "png"
    elif export_format == "JPEG":
        if save_img.mode in ("RGBA", "P"):
            save_img = save_img.convert("RGB")
        save_img.save(export_bytes, format="JPEG", quality=quality)
        mime = "image/jpeg"
        ext = "jpg"
    else:
        save_img.save(export_bytes, format="WEBP", quality=quality)
        mime = "image/webp"
        ext = "webp"

    with d_col3:
        st.write("")
        st.download_button(
            label=f"⬇️ Download Processed Image ({ext.upper()})",
            data=export_bytes.getvalue(),
            file_name=f"smart_image_studio_result.{ext}",
            mime=mime,
            use_container_width=True
        )

else:
    # Friendly landing state
    st.info("👆 Please upload an image to begin editing with Smart Image Processing Studio Pro.")
    st.markdown("""
    ### 🚀 Included Features:
    * **🤖 AI Background Removal**: Powered by `rembg` & ONNX U2Net model with instant transparent PNG output or custom background color.
    * **🔬 Scientific Enhancements**: CLAHE local contrast equalizer, Total Variation Denoising, and Otsu adaptive document scanner (`scikit-image`).
    * **🎨 Artistic Effects**: Pencil Sketch, Color Sketch, Cartoonizer, Sepia tone, and Vignette shading (`opencv`).
    * **👁️ Computer Vision**: Real-time multi-scale Face & Eye detection with bounding boxes and detection counts.
    * **📊 Image Analytics**: Live RGB color distribution histograms and K-Means dominant color palette extraction (`matplotlib`).
    * **💾 Multi-Step Studio**: State-managed edit pipeline with **Apply to Canvas**, **Undo**, and **Reset** features.
    """)