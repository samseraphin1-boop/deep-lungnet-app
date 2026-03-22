import streamlit as st
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gdown
import os

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

st.title("Lung Cancer Detection System")
st.write("AI-based prediction using Clinical Data & Medical Images")

# ================================
# SIDEBAR
# ================================
option = st.sidebar.radio(
    "Select Prediction Mode",
    ["Clinical Data Prediction", "Image Prediction"]
)

# ================================
# 🔹 CLINICAL DATA PREDICTION
# ================================
if option == "Clinical Data Prediction":

    st.header("📊 Clinical Data Prediction")

    @st.cache_resource
    def load_clinical():
        model = pickle.load(open("best_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        features = pickle.load(open("feature_list.pkl", "rb"))
        return model, scaler, features

    model, scaler, features = load_clinical()

    st.write("Enter patient details:")

    user_input = {}

    for feature in features:
        user_input[feature] = st.number_input(feature, value=0.0)

    input_data = np.array([list(user_input.values())])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Clinical Outcome"):

        prediction = model.predict(input_scaled)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0][1]
            st.info(f"🔮 Survival Probability: {prob:.2f}")

        if prediction[0] == 1:
            st.success("✅ Patient is likely to SURVIVE")
        else:
            st.error("⚠️ Patient is NOT likely to survive")


# ================================
# 🔹 IMAGE PREDICTION (CNN - FIXED)
# ================================
elif option == "Image Prediction":

    st.header("🖼️ Lung Cancer Image Detection")

    # ===== MODEL (2 CLASS FIXED) =====
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(64 * 56 * 56, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # ✅ FIXED TO 2 CLASSES
            )

        def forward(self, x):
            return self.model(x)

    # ===== LOAD MODEL FROM GOOGLE DRIVE =====
    @st.cache_resource
    def load_image_model():
        model_path = "lung_cancer_model.pth"

        if not os.path.exists(model_path):
            with st.spinner("⬇️ Downloading AI Model..."):
                url = "https://drive.google.com/uc?id=1ynBvW6ji-0rrV1aWCknJECj9lps1GB8F"
                gdown.download(url, model_path, quiet=False)

        model = CNNModel()
        state_dict = torch.load(model_path, map_location="cpu")

        # Avoid crash if mismatch
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    model = load_image_model()

    # ✅ FIXED CLASSES (2 ONLY)
    classes = ["Normal", "Cancer"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    uploaded_file = st.file_uploader("Upload Lung Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = transform(image).unsqueeze(0)

        if st.button("Predict Image"):

            with st.spinner("🔍 Analyzing Image..."):
                outputs = model(img)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                class_name = classes[predicted.item()]
                confidence = probs[0][predicted.item()].item()

            st.success(f"🧠 Prediction: {class_name}")
            st.info(f"Confidence: {confidence:.2f}")

            # Explanation
            if class_name == "Cancer":
                st.error("⚠️ Cancer detected. Please consult a medical professional.")
            else:
                st.success("✅ No cancer detected. Lung appears normal.")
