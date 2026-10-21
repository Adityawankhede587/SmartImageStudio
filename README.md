# 🖼️ Smart Image Processing Studio Pro

An interactive, multi-purpose computer vision and image processing studio built with **Streamlit**, **OpenCV**, **Pillow**, **scikit-image**, **rembg**, and **Matplotlib**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.24%2B-orange.svg)](https://scikit-image.org/)
[![rembg](https://img.shields.io/badge/AI_Background_Removal-rembg-green.svg)](https://github.com/danielgatis/rembg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

**Smart Image Processing Studio Pro** transforms raw images into professional assets with a unified workflow. It bridges modern AI background segmentation, advanced scientific image filtering, computer vision diagnostics, and artistic rendering inside a reactive, responsive web application.

---

## 🚀 Key Features

### 🤖 1. AI Background Removal (`rembg` + `onnxruntime`)
- High-precision automated foreground extraction powered by deep learning U2Net models.
- **Export Options**: Transparent PNG, clean solid white/black backdrops, or custom HEX color fills with fine alpha matting.

### 🔬 2. Scientific & Restoration Suite (`scikit-image`)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization to uncover hidden shadow details without over-amplifying noise.
- **Total Variation Denoising**: Smooths out high-frequency sensor noise and digital grain while retaining sharp object edges.
- **Otsu Document Binarization**: Computes optimal clustering thresholds for clean black-and-white document scans.
- **Local Adaptive Thresholding**: Normalizes uneven illumination gradients across text and diagrams.

### 👁️ 3. Computer Vision & Feature Detection (`OpenCV`)
- **Multi-Scale Face & Eye Detection**: Haar Cascade classification with bounding boxes, custom color palettes, and real-time detection counters.
- **Canny Edge Detection**: Dual-threshold gradient edge detection with adjustable sensitivity.

### 🎨 4. Artistic & Stylization Suite
- **Pencil Sketch**: Converts photos into authentic grayscale or colored pencil drawings.
- **Cartoonizer**: Edge-preserving bilateral color quantization combined with adaptive edge sketches.
- **Sepia Vintage Tone**: Warm nostalgic retro film simulation.
- **Vignette**: Radial Gaussian falloff shading.
- **Color Inversion**: Negative spectrum transformations.

### ⚙️ 5. Geometry, Lighting & Watermarking
- **Geometry**: Aspect-ratio locked resizing, continuous $\pm 180^\circ$ rotation with auto-expanding canvas, and bilateral flips.
- **Lighting Controls**: Fine-grained Brightness, Contrast, and Color Saturation enhancers.
- **Dynamic Watermarking**: Configurable text, font scaling, color picker, and dynamic preset alignments (*Bottom-Right*, *Center*, *Top-Left*, etc.).

### 📊 6. Analytics & Palette Extraction (`Matplotlib`)
- **Live RGB Histograms**: Visualizes red, green, and blue channel intensity distributions.
- **Dominant Color Extraction**: K-Means clustering generates the 5 dominant colors with HEX color badges and RGB values.

### 💾 7. Stateful Multi-Step Studio
- **Edit Chaining**: Apply current adjustments permanently to the canvas and continue stacking new filters.
- **Undo / Reset**: Safe history stack with one-click undo and full reset to original.
- **Multi-Format Downloads**: Export in **PNG**, **JPEG** (with quality compression slider), or **WEBP**.

---

## 📁 Repository Structure

```
SmartImageStudio/
├── app.py                # Main Streamlit web application & processing pipeline
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation & usage guide
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Adityawankhede587/SmartImageStudio.git
cd SmartImageStudio
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the studio using Streamlit:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧰 Tech Stack

- **UI & Web Engine**: [Streamlit](https://streamlit.io/)
- **AI Background Segmentation**: [rembg](https://github.com/danielgatis/rembg) & [ONNX Runtime](https://onnxruntime.ai/)
- **Scientific Image Processing**: [scikit-image](https://scikit-image.org/) & [SciPy](https://scipy.org/)
- **Computer Vision**: [OpenCV (cv2)](https://opencv.org/)
- **Image Primitives & Typography**: [Pillow (PIL)](https://pillow.readthedocs.io/)
- **Data Analytics & Plotting**: [Matplotlib](https://matplotlib.org/) & [NumPy](https://numpy.org/)

---

## 📄 License

This project is licensed under the MIT License.
