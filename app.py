import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
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
# DOWNLOAD MODEL FROM DRIVE
# -----------------------------
FILE_ID = "1m_99ziaptnbqhl0vOhLyq9dXMiLtbh-L"

if not os.path.exists("cnn_lung_model.pth"):
    with st.spinner("Downloading CNN model..."):
        url = f"https://drive.google.com/uc?id={1m_99ziaptnbqhl0vOhLyq9dXMiLtbh-L}"
        gdown.download(url, "cnn_lung_model.pth", quiet=False)

# -----------------------------
# LOAD CNN MODEL (STATE_DICT FIX)
# -----------------------------
@st.cache_resource
def load_cnn():
    model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 3)
    )

    state_dict = torch.load("cnn_lung_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

cnn_model = load_cnn()

# -----------------------------
# LOAD ML MODEL
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

        with st.spinner("Analyzing..."):
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
            st.error("ML model not found!")
        else:
            input_data = np.array([[age, smoking, anxiety, fatigue]])

            pred = ml_model.predict(input_data)

            if pred[0] == 1:
                st.error("⚠️ High Risk of Lung Cancer")
            else:
                st.success("✅ Low Risk")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Developed using CNN + Machine Learning | Streamlit 🚀")
