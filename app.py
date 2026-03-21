import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

st.title("🫁 Lung Cancer Detection System")
st.markdown("### Hybrid AI Model (CNN + Machine Learning)")

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
    with open("best_ml_model.pkl", "rb") as f:
        return pickle.load(f)

ml_model = load_ml()

classes = ["Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

# -----------------------------
# SIDEBAR
# -----------------------------
option = st.sidebar.radio("Choose Prediction Type", ["Image Prediction", "Clinical Prediction"])

# -----------------------------
# IMAGE SECTION
# -----------------------------
if option == "Image Prediction":

    st.header("🖼️ Image-Based Lung Cancer Detection")

    st.info("""
    📌 Instructions:
    - Upload lung histopathology image only
    - Image should be similar to dataset type
    - Format: JPG / PNG
    - Avoid random or unrelated images
    """)

    st.subheader("📷 Example Input Image")
    st.image("sample_lung_image.png", caption="Reference Image", use_container_width=True)

    uploaded_file = st.file_uploader("Upload Lung Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(f"Prediction: {classes[pred]}")

        st.subheader("Confidence Scores")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {probs[0][i]:.4f}")

# -----------------------------
# CLINICAL SECTION
# -----------------------------
if option == "Clinical Prediction":

    st.header("📊 Clinical Data Prediction")

    st.info("Fill the details below to predict lung cancer risk")

    age = st.number_input("Age", 1, 100)
    smoking = st.selectbox("Smoking", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])

    if st.button("Predict"):
        input_data = np.array([[age, smoking, anxiety, fatigue]])

        pred = ml_model.predict(input_data)

        if pred[0] == 1:
            st.error("⚠️ High Risk of Lung Cancer")
        else:
            st.success("✅ Low Risk / No Cancer")
