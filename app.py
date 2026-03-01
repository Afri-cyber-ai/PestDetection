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

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🐛 Agricultural Pest Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of your crops/pests and click Analyze to detect pests.</p>", unsafe_allow_html=True)
st.markdown("---")

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
    # Open and display uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # Add Analyze Button
    if st.button("Analyze to Detect Pests"):
        with st.spinner("Analyzing image, please wait..."):
            results = model(image_np)
            annotated_image = results[0].plot()

        st.subheader("Detected Pests")
        st.image(annotated_image, use_column_width=True)
