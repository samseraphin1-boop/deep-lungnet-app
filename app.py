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
CNN_ID = "1m_99ziaptnbqhl0vOhLyq9dXMiLtbh-L"   # your CNN model
ML_ID = "YOUR_ML_FILE_ID"  # replace if using ML model from drive

# -----------------------------
# DOWNLOAD FUNCTION
# -----------------------------
def download_file(file_id, output):
    if file_id != "YOUR_ML_FILE_ID":  # skip empty id
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)

# -----------------------------
# DOWNLOAD MODELS
# -----------------------------
download_file(CNN_ID, "cnn_lung_model.pth")

# Uncomment if ML model is in drive
# download_file(ML_ID, "best_ml_model.pkl")

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
# LOAD ML MODEL (OPTIONAL)
# -----------------------------
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

        if ml_model is None:
            st.error("ML model not loaded!")
        else:
            input_data = np.array([[age, smoking, anxiety, fatigue]])

            pred = ml_model.predict(input_data)

            if pred[0] == 1:
                st.error("⚠️ High Risk of Lung Cancer")
            else:
                st.success("✅ Low Risk / No Cancer")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Developed using CNN + ML | Streamlit 🚀")
