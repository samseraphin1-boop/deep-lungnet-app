import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import gdown
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="DeepLungNet", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
# 🫁 DeepLungNet
### AI-Powered Lung Cancer Detection System
""")

st.write("Hybrid Model using CNN (Image) + ML (Clinical Data)")

# -----------------------------
# DOWNLOAD MODELS (GOOGLE DRIVE)
# -----------------------------
if not os.path.exists("cnn_lung_model.pth"):
    with st.spinner("🔄 Downloading CNN Model..."):
        url = "https://drive.google.com/uc?id=1vS7yE3I0wgVxFIoDJuMZAjwYXn0w_pEP"
        gdown.download(url, "cnn_lung_model.pth", quiet=False)

# (Optional ML model download if needed)
# Add your ML file ID if large

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_cnn():
    model = torch.load("cnn_lung_model.pth", map_location="cpu")
    model.eval()
    return model

cnn_model = load_cnn()

@st.cache_resource
def load_ml():
    try:
        with open("best_ml_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

ml_model = load_ml()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = ["Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Choose Prediction Type", ["Image Prediction", "Clinical Prediction"])

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
if option == "Image Prediction":

    st.header("🖼️ Image-Based Detection")

    st.info("""
📌 Upload lung histopathology image  
✔ Format: JPG / PNG  
✔ Similar to dataset (microscope images)  
❌ Avoid random images
""")

    st.subheader("📷 Example Input")
    st.image("sample_lung_image.png", use_container_width=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        img_tensor = transform(image).unsqueeze(0)

        with st.spinner("🔍 Analyzing Image..."):
            with torch.no_grad():
                outputs = cnn_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

        with col2:
            st.success(f"🧠 Prediction: {classes[pred]}")

            st.subheader("📊 Confidence Scores")

            for i, cls in enumerate(classes):
                st.progress(float(probs[0][i]))
                st.write(f"{cls}: {probs[0][i]:.4f}")

# -----------------------------
# CLINICAL PREDICTION
# -----------------------------
if option == "Clinical Prediction":

    st.header("📊 Clinical Data Prediction")

    st.info("Enter patient details below")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100)
        smoking = st.selectbox("Smoking", [0, 1])
        anxiety = st.selectbox("Anxiety", [0, 1])

    with col2:
        fatigue = st.selectbox("Fatigue", [0, 1])
        peer_pressure = st.selectbox("Peer Pressure", [0, 1])
        chronic = st.selectbox("Chronic Disease", [0, 1])

    if st.button("Predict"):

        if ml_model is None:
            st.error("ML model not loaded!")
        else:
            input_data = np.array([[age, smoking, anxiety, fatigue, peer_pressure, chronic]])

            with st.spinner("🔍 Predicting..."):
                pred = ml_model.predict(input_data)

            if pred[0] == 1:
                st.error("⚠️ High Risk of Lung Cancer")
            else:
                st.success("✅ Low Risk / No Cancer")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Developed using Deep Learning & Machine Learning | Streamlit App 🚀")
