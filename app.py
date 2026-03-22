import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="DEEPLUNGNET", page_icon="🫁", layout="wide")

# --- VIBRANT CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #d32f2f; color: white; border-radius: 8px; height: 3em; font-weight: bold; }
    .stButton>button:hover { background-color: #b71c1c; border: 1px solid white; }
    h1 { color: #d32f2f; text-align: center; font-family: 'Trebuchet MS'; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CNN MODEL ARCHITECTURE ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_assets():
    with open('best_model.pkl', 'rb') as f: ml_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('feature_list.pkl', 'rb') as f: features = pickle.load(f)
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load('lung_cancer_model.pth', map_location=torch.device('cpu')))
    cnn_model.eval()
    return ml_model, scaler, features, cnn_model

ml_model, scaler, ml_features, cnn_model = load_assets()

# --- UI LAYOUT ---
st.markdown("<h1>DEEPLUNGNET: LUNG CANCER ANALYSIS SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Integrating Machine Learning & CNN for Detection and Survival Prediction</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ CNN Image Detection", "📊 Clinical Survival Prediction"])

with tab1:
    st.header("CT-Scan Image Classification")
    up_file = st.file_uploader("Upload Scanning Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if up_file:
        img = Image.open(up_file).convert('RGB')
        st.image(img, caption="Target Image", width=400)
        if st.button("RUN CNN DIAGNOSIS"):
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                prob = cnn_model(img_t).item()
            if prob > 0.5:
                st.error(f"⚠️ POSITIVE: Lung Cancer Detected (Prob: {prob:.2%})")
            else:
                st.success(f"✅ NEGATIVE: No Cancer Detected (Prob: {1-prob:.2%})")

with tab2:
    st.header("Clinical Outcome Prediction")
    c1, c2 = st.columns(2)
    with c1:
        yrs = st.slider("Years of Smoking", 0, 60, 10)
        air = st.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"])
        pain = st.radio("Chest Pain?", ["No", "Yes"])
    with c2:
        stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        treat = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation", "Palliative Care"])

    if st.button("PREDICT SURVIVAL CHANCE"):
        # Map inputs to model features
        feat_dict = {f: 0 for f in ml_features}
        if 'Years_Smoking' in feat_dict: feat_dict['Years_Smoking'] = yrs
        if air == "Low" and 'Air_Pollution_Exposure_Low' in feat_dict: feat_dict['Air_Pollution_Exposure_Low'] = 1
        if pain == "Yes" and 'Chest_Pain_Yes' in feat_dict: feat_dict['Chest_Pain_Yes'] = 1
        if f'Cancer_Stage_{stage}' in feat_dict: feat_dict[f'Cancer_Stage_{stage}'] = 1
        if f'Treatment_{treat}' in feat_dict: feat_dict[f'Treatment_{treat}'] = 1
        
        input_df = pd.DataFrame([feat_dict])[ml_features]
        scaled_input = scaler.transform(input_df)
        res = ml_model.predict(scaled_input)[0]
        prob_surv = ml_model.predict_proba(scaled_input)[0][1]
        
        if res == 1:
            st.success(f"High Survival Probability: {prob_surv:.2%}")
        else:
            st.warning(f"Moderate/Low Survival Probability: {prob_surv:.2%}")

st.sidebar.markdown("### Project Info")
st.sidebar.write("DEEPLUNGNET v1.0")
st.sidebar.info("This system uses a Hybrid CNN-ML approach for lung cancer screening.")
