import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2
import numpy as np
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Image Processing Studio",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🖼️ Smart Image Processing Studio")

st.write(
    "An interactive image processing application using "
    "Streamlit, Pillow and OpenCV."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Image Processing")

operation = st.sidebar.selectbox(
    "Select Operation",
    [
        "Original",
        "Resize",
        "Blur",
        "Sharpen",
        "Brightness",
        "Contrast",
        "Add Text",
        "Grayscale",
        "Edge Detection",
        "Face Detection"
    ]
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Create a copy for processing
    processed_image = original_image.copy()


    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    if operation == "Original":

        processed_image = original_image.copy()


    # ========================================================
    # 2. RESIZE
    # ========================================================

    elif operation == "Resize":

        st.sidebar.subheader("Resize Settings")

        width = st.sidebar.number_input(
            "Width",
            min_value=50,
            max_value=3000,
            value=original_image.width,
            step=10
        )

        height = st.sidebar.number_input(
            "Height",
            min_value=50,
            max_value=3000,
            value=original_image.height,
            step=10
        )

        processed_image = original_image.resize(
            (width, height)
        )


    # ========================================================
    # 3. BLUR
    # ========================================================

    elif operation == "Blur":

        st.sidebar.subheader("Blur Settings")

        blur_amount = st.sidebar.slider(
            "Blur Amount",
            min_value=1,
            max_value=20,
            value=5
        )

        processed_image = original_image.filter(
            ImageFilter.GaussianBlur(
                blur_amount
            )
        )


    # ========================================================
    # 4. SHARPEN
    # ========================================================

    elif operation == "Sharpen":

        processed_image = original_image.filter(
            ImageFilter.SHARPEN
        )


    # ========================================================
    # 5. BRIGHTNESS
    # ========================================================

    elif operation == "Brightness":

        st.sidebar.subheader("Brightness Settings")

        brightness = st.sidebar.slider(
            "Brightness",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1
        )

        enhancer = ImageEnhance.Brightness(
            original_image
        )

        processed_image = enhancer.enhance(
            brightness
        )


    # ========================================================
    # 6. CONTRAST
    # ========================================================

    elif operation == "Contrast":

        st.sidebar.subheader("Contrast Settings")

        contrast = st.sidebar.slider(
            "Contrast",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1
        )

        enhancer = ImageEnhance.Contrast(
            original_image
        )

        processed_image = enhancer.enhance(
            contrast
        )


    # ========================================================
    # 7. ADD TEXT
    # ========================================================

    elif operation == "Add Text":

        st.sidebar.subheader("Text Settings")

        text = st.sidebar.text_input(
            "Enter Text",
            value="Smart Image Studio"
        )

        x_position = st.sidebar.number_input(
            "X Position",
            min_value=0,
            max_value=max(
                0,
                original_image.width - 1
            ),
            value=20
        )

        y_position = st.sidebar.number_input(
            "Y Position",
            min_value=0,
            max_value=max(
                0,
                original_image.height - 1
            ),
            value=20
        )

        processed_image = original_image.copy()

        draw = ImageDraw.Draw(
            processed_image
        )

        draw.text(
            (
                x_position,
                y_position
            ),
            text,
            fill="red"
        )


    # ========================================================
    # 8. GRAYSCALE - OPENCV
    # ========================================================

    elif operation == "Grayscale":

        # Pillow image → NumPy array
        image_array = np.array(
            original_image
        )

        # RGB → Grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Grayscale → RGB
        gray_rgb = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB
        )

        # NumPy → Pillow
        processed_image = Image.fromarray(
            gray_rgb
        )


    # ========================================================
    # 9. EDGE DETECTION - OPENCV
    # ========================================================

    elif operation == "Edge Detection":

        # Convert Pillow image to NumPy
        image_array = np.array(
            original_image
        )

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Detect edges
        edges = cv2.Canny(
            gray,
            100,
            200
        )

        # Convert back to Pillow
        processed_image = Image.fromarray(
            edges
        )


    # ========================================================
    # 10. FACE DETECTION - OPENCV
    # ========================================================

    elif operation == "Face Detection":

        # Convert Pillow → NumPy
        image_array = np.array(
            original_image
        )

        # Convert RGB → BGR
        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Convert image to grayscale
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # Haar Cascade classifier
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangle around every detected face
        for (x, y, w, h) in faces:

            cv2.rectangle(
                image_bgr,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                image_bgr,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Convert BGR → RGB
        result_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # NumPy → Pillow
        processed_image = Image.fromarray(
            result_rgb
        )

        # Display face count
        st.success(
            f"👤 Faces detected: {len(faces)}"
        )


    # ========================================================
    # PROCESSED IMAGE SECTION
    # ========================================================

    st.subheader("🖼️ Processed Image")

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.markdown("### Original Image")

        st.image(
            original_image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PROCESSED IMAGE
    # --------------------------------------------------------

    with col2:

        st.markdown("### Processed Result")

        st.image(
            processed_image,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader("⬇️ Download")

    image_bytes = io.BytesIO()

    processed_image.save(
        image_bytes,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Processed Image",
        data=image_bytes.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )


# ============================================================
# NO IMAGE MESSAGE
# ============================================================

else:

    st.info(
        "👆 Please upload an image to start "
        "using Smart Image Processing Studio."
    )