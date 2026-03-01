import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

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
    # Open image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Create two columns
    col1, col2 = st.columns(2)

    # Show original image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Run YOLO inference
    results = model(image_np)

    # Get annotated image
    annotated_image = results[0].plot()

    # Convert BGR to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Show detected image
    with col2:
        st.subheader("Detected Pests")
        st.image(annotated_image, use_column_width=True)
