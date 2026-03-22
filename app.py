import streamlit as st
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

st.title("🫁 Lung Cancer Detection System")
st.write("Choose prediction type from sidebar")

# ================================
# SIDEBAR
# ================================
option = st.sidebar.radio(
    "Select Mode",
    ["Clinical Data Prediction", "Image Prediction"]
)

# ================================
# ================================
# 🔹 1. CLINICAL DATA MODEL
# ================================
# ================================
if option == "Clinical Data Prediction":

    st.header("📊 Clinical Data Prediction")

    # Load model files
    model = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))

    st.write("Enter patient details:")

    user_input = {}

    # Create inputs dynamically
    for feature in features:
        user_input[feature] = st.number_input(feature, value=0.0)

    input_data = np.array([list(user_input.values())])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Clinical Outcome"):

        prediction = model.predict(input_scaled)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0][1]
            st.write(f"🔮 Survival Probability: {prob:.2f}")

        if prediction[0] == 1:
            st.success("✅ Patient is likely to SURVIVE")
        else:
            st.error("⚠️ Patient is NOT likely to survive")


# ================================
# ================================
# 🔹 2. IMAGE MODEL (CNN)
# ================================
# ================================
elif option == "Image Prediction":

    st.header("🖼️ Lung Cancer Image Detection")

    # ===== Define Model (SAME AS TRAINING) =====
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 56 * 56, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # 3 classes
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x


    # Load model
    model = CNNModel()
    model.load_state_dict(torch.load("lung_cancer_model.pth", map_location="cpu"))
    model.eval()

    # Class labels
    classes = ["Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

    # Image transform
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

            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

            class_name = classes[predicted.item()]

            st.success(f"🧠 Prediction: {class_name}")

            # Explanation
            if class_name == "Adenocarcinoma":
                st.warning("Most common lung cancer type, starts in outer lung cells.")
            elif class_name == "Squamous Cell Carcinoma":
                st.warning("Linked to smoking, occurs in central lungs.")
            else:
                st.success("Normal lung tissue detected.")
