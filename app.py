import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Agricultural Pest Detection",
    page_icon="🐛",
    layout="wide"
)

st.title("🐛 Agricultural Pest Detection App")
st.write("Upload an image to detect agricultural pests using YOLOv8.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("PestDetectionModel.pt")
    return model

model = load_model()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Run YOLO inference
    results = model(image_np)

    # Get annotated image directly
    annotated_image = results[0].plot()

    with col2:
        st.subheader("Detected Pests")
        st.image(annotated_image, use_column_width=True)
