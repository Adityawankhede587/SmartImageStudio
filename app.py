import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2
import numpy as np
import io




# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Image Processing Studio",
    page_icon="🖼️",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🖼️ Smart Image Processing Studio")

st.write(
    "An interactive image processing application using "
    "Streamlit, Pillow and OpenCV."
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Image Processing")

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


# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------
# PROCESS IMAGE
# ---------------------------------------------------

if uploaded_file is not None:

    # Open image using Pillow
    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")

    st.image(
        original_image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ------------------------------------------------
    # PROCESSING
    # ------------------------------------------------

    processed_image = original_image.copy()


    # ================================================
    # 1. ORIGINAL
    # ================================================

    if operation == "Original":

        processed_image = original_image


    # ================================================
    # 2. RESIZE
    # ================================================

    elif operation == "Resize":

        width = st.sidebar.slider(
            "Width",
            100,
            1500,
            original_image.width
        )

        height = st.sidebar.slider(
            "Height",
            100,
            1500,
            original_image.height
        )

        processed_image = original_image.resize(
            (width, height)
        )


    # ================================================
    # 3. BLUR
    # ================================================

    elif operation == "Blur":

        blur_amount = st.sidebar.slider(
            "Blur Amount",
            1,
            20,
            5
        )

        processed_image = original_image.filter(
            ImageFilter.GaussianBlur(blur_amount)
        )


    # ================================================
    # 4. SHARPEN
    # ================================================

    elif operation == "Sharpen":

        processed_image = original_image.filter(
            ImageFilter.SHARPEN
        )


    # ================================================
    # 5. BRIGHTNESS
    # ================================================

    elif operation == "Brightness":

        brightness = st.sidebar.slider(
            "Brightness",
            0.1,
            3.0,
            1.0,
            0.1
        )

        enhancer = ImageEnhance.Brightness(
            original_image
        )

        processed_image = enhancer.enhance(
            brightness
        )


    # ================================================
    # 6. CONTRAST
    # ================================================

    elif operation == "Contrast":

        contrast = st.sidebar.slider(
            "Contrast",
            0.1,
            3.0,
            1.0,
            0.1
        )

        enhancer = ImageEnhance.Contrast(
            original_image
        )

        processed_image = enhancer.enhance(
            contrast
        )


    # ================================================
    # 7. ADD TEXT
    # ================================================

    elif operation == "Add Text":

        text = st.sidebar.text_input(
            "Enter Text",
            "Smart Image Studio"
        )

        x = st.sidebar.slider(
            "X Position",
            0,
            max(0, original_image.width - 1),
            20
        )

        y = st.sidebar.slider(
            "Y Position",
            0,
            max(0, original_image.height - 1),
            20
        )

        processed_image = original_image.copy()

        draw = ImageDraw.Draw(processed_image)

        draw.text(
            (x, y),
            text,
            fill="red"
        )


    # ================================================
    # 8. GRAYSCALE - OPENCV
    # ================================================

    elif operation == "Grayscale":

        # Pillow → NumPy
        image_array = np.array(original_image)

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

        processed_image = Image.fromarray(
            gray_rgb
        )


    # ================================================
    # 9. EDGE DETECTION - OPENCV
    # ================================================

    elif operation == "Edge Detection":

        image_array = np.array(original_image)

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        processed_image = Image.fromarray(
            edges
        )


    # ================================================
    # 10. FACE DETECTION - OPENCV
    # ================================================

    elif operation == "Face Detection":

        image_array = np.array(original_image)

        # RGB → BGR
        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Convert to grayscale
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangle around faces
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

        # BGR → RGB
        result_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        processed_image = Image.fromarray(
            result_rgb
        )

        st.info(
            f"Number of faces detected: {len(faces)}"
        )


    # ------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------

    st.subheader("Processed Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original_image,
            caption="Original",
            use_container_width=True
        )

    with col2:

        st.image(
            processed_image,
            caption="Processed",
            use_container_width=True
        )


    # ------------------------------------------------
    # DOWNLOAD RESULT
    # ------------------------------------------------

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

else:

    st.info(
        "👆 Please upload an image to start."
    )