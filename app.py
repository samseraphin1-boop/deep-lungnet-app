import streamlit as st
import torch
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

st.title("🫁 DeepLungNet")
st.write("AI-Based Lung Cancer Detection System")

# -----------------------------
# GOOGLE DRIVE FILE IDS
# -----------------------------
CNN_ID = "YOUR_CNN_FILE_ID"
ML_ID = "YOUR_ML_FILE_ID"
SCALER_ID = "YOUR_SCALER_FILE_ID"

# -----------------------------
# DOWNLOAD FILES
# -----------------------------
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Download models
download_file(CNN_ID, "cnn_lung_model.pth")
download_file(ML_ID, "best_ml_model.pkl")
download_file(SCALER_ID, "scaler.pkl")

# -----------------------------
# LOAD CNN MODEL (FULL MODEL)
# -----------------------------
@st.cache_resource
def load_cnn():
    model = torch.load("cnn_lung_model.pth", map_location="cpu")
    model.eval()
    return model

cnn_model = load_cnn()

# -----------------------------
# LOAD ML MODEL + SCALER
# -----------------------------
@st.cache_resource
def load_ml():
    with open("best_ml_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

ml_model, scaler = load_ml()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = ["Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

# -----------------------------
# SIDEBAR
# -----------------------------
option = st.sidebar.radio("Choose Prediction Type", ["Image", "Clinical"])

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
if option == "Image":

    st.header("🖼️ Image Prediction")

    uploaded_file = st.file_uploader("Upload Lung Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        img_tensor = transform(image).unsqueeze(0)

        with st.spinner("Analyzing Image..."):
            with torch.no_grad():
                outputs = cnn_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

        st.success(f"Prediction: {classes[pred]}")

        st.subheader("Confidence Scores")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {probs[0][i]:.4f}")

# -----------------------------
# CLINICAL PREDICTION
# -----------------------------
if option == "Clinical":

    st.header("📊 Clinical Prediction")

    age = st.number_input("Age", 1, 100)
    smoking = st.selectbox("Smoking", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])

    if st.button("Predict"):

        input_data = np.array([[age, smoking, anxiety, fatigue]])
        input_scaled = scaler.transform(input_data)

        pred = ml_model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ High Risk of Lung Cancer")
        else:
            st.success("✅ Low Risk / No Cancer")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Developed using CNN + ML | Streamlit 🚀")
