"""
TASK 4 - Streamlit App: Clinical Breast Cancer Risk Predictor
Developer: Zikra
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer

# 1. Page Configuration
st.set_page_config(
    page_title="Clinical Breast Cancer Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# 2. Sidebar Settings & Theme Management
st.sidebar.header("⚙️ Settings & Controls")
theme_choice = st.sidebar.radio("Choose Theme", ["Light Theme", "Dark Theme"])

if theme_choice == "Dark Theme":
    bg_color = "#1E1E1E"
    text_color = "#FFFFFF"
    card_bg = "#2D2D2D"
    border_color = "#444444"
    sidebar_bg = "#252526"
    btn_bg = "#8E24AA"
    btn_text = "#FFFFFF"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    card_bg = "#F8F9FA"
    border_color = "#E0E0E0"
    sidebar_bg = "#F0F2F6"
    btn_bg = "#8E24AA"
    btn_text = "#FFFFFF"

# Custom CSS Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    .main-title {{
        font-size: 34px;
        font-weight: bold;
        color: #8E24AA;
        text-align: center;
        margin-bottom: 25px;
    }}
    .card-box {{
        background-color: {card_bg};
        padding: 22px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-bottom: 20px;
    }}
    .summary-text-box {{
        background-color: {card_bg};
        border-left: 5px solid #8E24AA;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        border-top: 1px solid {border_color};
        border-right: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
    }}
    .stButton>button {{
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
    }}
    .footer-card {{
        background: linear-gradient(135deg, #6A1B9A, #8E24AA);
        color: #FFFFFF !important;
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        margin-top: 40px;
        font-size: 16px;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .footer-card b {{
        color: #FFFFFF !important;
    }}
    </style>
""", unsafe_allow_html=True)

# 3. Main Title
st.markdown('<div class="main-title">🩺 Clinical Breast Cancer Risk Predictor</div>', unsafe_allow_html=True)

# 4. Load Model and Dataset
@st.cache_resource
def load_model_data():
    saved = joblib.load("clinical_model.joblib")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return saved["model"], saved["features"], df

model, features, df = load_model_data()
defaults = df.mean().to_dict()

key_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

# Labels with Medical Measurement Units
feature_units = {
    "mean radius": "Mean Radius (mm)",
    "mean texture": "Mean Texture (gray-scale value)",
    "mean perimeter": "Mean Perimeter (mm)",
    "mean area": "Mean Area (mm²)",
    "mean smoothness": "Mean Smoothness (local variation)"
}

malignant_sample = {
    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.80,
    "mean area": 1001.0, "mean smoothness": 0.1184
}
benign_sample = {
    "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46,
    "mean area": 566.30, "mean smoothness": 0.09779
}

# 5. Sidebar Setup
st.sidebar.markdown("---")
st.sidebar.header("🎯 Quick Sample Data")

if st.sidebar.button("🔴 Load High Risk Sample"):
    for feat in key_features:
        st.session_state[f"slider_{feat}"] = malignant_sample[feat]
    st.session_state.analyzed = True

if st.sidebar.button("🟢 Load Low Risk Sample"):
    for feat in key_features:
        st.session_state[f"slider_{feat}"] = benign_sample[feat]
    st.session_state.analyzed = True

st.sidebar.markdown("---")
st.sidebar.header("📋 Feature Guide")
st.sidebar.markdown("""
* **Mean Radius (mm):** Mean distance from center to points on the perimeter.
* **Mean Texture:** Standard deviation of gray-scale values.
* **Mean Perimeter (mm):** Total boundary size of the core tumor.
* **Mean Area (mm²):** Total surface area occupied by the tumor.
* **Mean Smoothness:** Local variation in radius lengths.
""")

# 6. Main Interface Layout
col1, col2 = st.columns([1.1, 1])

# Calculate Prediction Values First to use across columns
full_input = defaults.copy()
for feat in key_features:
    full_input[feat] = st.session_state.get(f"slider_{feat}", float(df[feat].mean()))

input_df = pd.DataFrame([full_input])[features]
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
malignant_prob = proba[0] * 100
benign_prob = proba[1] * 100

# --- LEFT COLUMN: Input Controls & Tumor Shape ---
with col1:
    st.subheader("📋 Tumor Measurements")
    
    current_input = {}
    for feat in key_features:
        default_val = float(df[feat].mean())
        val = st.slider(
            feature_units[feat],
            float(df[feat].min()),
            float(df[feat].max()),
            default_val,
            key=f"slider_{feat}"
        )
        current_input[feat] = val

    analyze_btn = st.button("🩺 Analyze Risk", use_container_width=True)

    # 🖼️ Tumor Shape Profile Placed Directly Under "Analyze Risk" Button
    if analyze_btn or st.session_state.get("analyzed", False):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🖼️ Simulated Anatomical Tumor Shape Profile")
        
        radius_val = st.session_state.get("slider_mean radius", current_input["mean radius"])
        smoothness_val = st.session_state.get("slider_mean smoothness", current_input["mean smoothness"])
        
        theta = np.linspace(0, 2*np.pi, 200)
        noise = np.sin(7*theta) * smoothness_val * (20 if prediction == 0 else 5)
        r = radius_val + noise
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        fig_shape = go.Figure()
        fig_shape.add_trace(go.Scatter(
            x=x, y=y, fill="toself",
            fillcolor="rgba(229, 57, 53, 0.4)" if prediction == 0 else "rgba(67, 160, 71, 0.4)",
            line=dict(color="#E53935" if prediction == 0 else "#43A047", width=3),
            name="Tumor Boundary"
        ))
        fig_shape.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_shape, use_container_width=True)

# --- RIGHT COLUMN: Diagnostic Results & Reports ---
with col2:
    st.subheader("📊 Diagnostic Analysis & Risk Report")

    if analyze_btn or st.session_state.get("analyzed", False):
        # 1. Main Prediction Card
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        if prediction == 0:
            st.error("### ⚠️ Prediction: Malignant (High Risk)")
            st.write("Clinical analysis indicates a high probability of malignant tumor characteristics.")
        else:
            st.success("### ✅ Prediction: Benign (Low Risk)")
            st.write("Clinical analysis indicates a non-cancerous benign tumor profile.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. Detailed Text Report
        st.subheader("📝 Detailed Text Report & Risk Summary")
        
        if 45 <= malignant_prob <= 55:
            explanation_text = f"""
            <b>Borderline Assessment Summary:</b><br>
            • <b>Malignant Risk:</b> {malignant_prob:.1f}%<br>
            • <b>Benign Likelihood:</b> {benign_prob:.1f}%<br><br>
            <b>Clinical Explanation:</b> The tumor measurements place this profile in a <i>borderline risk zone</i>. 
            There is an equal/near-equal probability between benign and malignant characteristics. 
            <b>Immediate Recommendation:</b> Secondary diagnostic evaluation (Biopsy / MRI) is strongly advised.
            """
        elif prediction == 0:
            explanation_text = f"""
            <b>High Risk Assessment Summary:</b><br>
            • <b>Malignant Risk (Cancerous Chance):</b> <span style="color:#E53935; font-weight:bold;">{malignant_prob:.1f}%</span><br>
            • <b>Benign Likelihood (Safety Chance):</b> {benign_prob:.1f}%<br><br>
            <b>Clinical Explanation:</b> Based on the entered tumor measurements, 
            the AI model estimates a <b>{malignant_prob:.1f}% risk</b> that this tumor exhibits malignant (cancerous) traits. 
            <b>Recommendation:</b> Immediate oncologist consultation and tissue biopsy are recommended.
            """
        else:
            explanation_text = f"""
            <b>Low Risk Assessment Summary:</b><br>
            • <b>Benign Likelihood (Safety Chance):</b> <span style="color:#43A047; font-weight:bold;">{benign_prob:.1f}%</span><br>
            • <b>Malignant Risk (Cancerous Chance):</b> {malignant_prob:.1f}%<br><br>
            <b>Clinical Explanation:</b> Based on the entered tumor measurements, 
            the AI model estimates a <b>{benign_prob:.1f}% likelihood</b> that this tumor is benign (non-cancerous). 
            The cancer risk is relatively low ({malignant_prob:.1f}%). 
            <b>Recommendation:</b> Regular clinical check-ups and routine follow-ups are suggested.
            """

        st.markdown(f'<div class="summary-text-box">{explanation_text}</div>', unsafe_allow_html=True)

        # 3. Graphical Probability Score Under Text Summary
        st.subheader("📈 Graphical Probability Score")
        fig = go.Figure(go.Bar(
            x=[malignant_prob, benign_prob],
            y=['Malignant Risk', 'Benign Likelihood'],
            orientation='h',
            marker=dict(color=['#E53935', '#43A047']),
            text=[f"{malignant_prob:.1f}%", f"{benign_prob:.1f}%"],
            textposition='auto'
        ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color, size=13)
        )
        st.plotly_chart(fig, use_container_width=True)

# 7. Redesigned & Prominent Footer
st.markdown("""
    <div class="footer-card">
        <h3 style="margin:0; font-size: 22px; color: #FFFFFF;">TASK 4: High-Imbalance Clinical Disease Predictor Pipeline</h3>
        <p style="margin: 8px 0 0 0; font-size: 16px;">
            <b>Develop by Zikra</b><br>
            BS Computer Science | Women University Mardan<br>
            Progree Machine Learning Internship<br>
            2026
        </p>
    </div>
""", unsafe_allow_html=True)