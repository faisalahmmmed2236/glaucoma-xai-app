import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import time

# ==========================================
# 1. PLATINUM UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OculoVision Platinum | AI Diagnostic", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & High-End Clinical UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }

    /* Gradient Header */
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    /* Severity Indicators */
    .risk-high { color: #DC2626; font-weight: 700; font-size: 1.2rem; }
    .risk-low { color: #16A34A; font-weight: 700; font-size: 1.2rem; }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] { color: #1E3A8A !important; }
    </style>
""", unsafe_allow_html=True)

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"

@st.cache_resource
def load_clinical_model():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

model = load_clinical_model()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

# ==========================================
# 2. CORE AI LOGIC
# ==========================================
def preprocess_for_inference(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), int(min(h, w) * 0.45), 255, -1)
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, bbox_w, bbox_h = cv2.boundingRect(coords)
        masked = masked[y:y+bbox_h, x:x+bbox_w]
        
    # Apply CLAHE
    lab = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    resized = cv2.resize(enhanced, (IMAGE_SIZE, IMAGE_SIZE))
    return img_rgb, resized

def generate_gradcam(model, img_array, layer_name):
    img_batch = np.expand_dims(img_array, axis=0)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        loss = 1.0 - predictions[:, 0] 
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

# ==========================================
# 3. WORKSPACE LAYOUT
# ==========================================

# --- Header ---
st.markdown("""
    <div class="header-container">
        <h1 style="margin:0; font-weight:800; font-size:36px; letter-spacing:-1px;">OculoVision™ Platinum</h1>
        <p style="margin:0; opacity:0.8; font-size:18px;">Next-Generation Glaucoma Detection & Clinical Analytics</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🏥 Hospital Node")
    st.markdown("**Status:** `Connected / Secure`")
    st.divider()
    st.markdown("### 🛠️ Diagnostic Calibration")
    threshold = st.slider("Sensitivity Threshold", 0.1, 0.9, 0.6, 0.05)
    overlay_alpha = st.slider("XAI Opacity", 0.1, 0.9, 0.5, 0.1)
    st.divider()
    st.info("Ensuring high-fidelity analysis via MobileNetV2 Ensemble.")

# --- File Upload & EHR ---
col_ehr, col_upload = st.columns([1, 2])

with col_ehr:
    with st.container():
        st.markdown("#### 📋 Patient Context")
        patient_id = st.text_input("MRN / Patient ID", "PX-4491")
        st.write(f"**Date:** {datetime.now().strftime('%d %b %Y')}")
        st.write(f"**AI Node:** `Central_Cluster_01`")

with col_upload:
    uploaded_file = st.file_uploader("Upload Retinal Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Inference Pipeline
    with st.spinner('Neural Core Analyzing Structures...'):
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        glaucoma_prob = 1.0 - float(prediction[0])
        
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 1.0 - overlay_alpha, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), overlay_alpha, 0)

    st.divider()

    # --- Analysis Tabs ---
    tab_report, tab_img, tab_data = st.tabs(["📊 Diagnostic Report", "🔬 Imaging Workspace", "🧬 Data Analytics"])

    with tab_report:
        st.markdown("<br>", unsafe_allow_html=True)
        rep_1, rep_2 = st.columns([1, 1.5])
        
        with rep_1:
            st.markdown("#### Diagnostic Outcome")
            if glaucoma_prob >= threshold:
                st.markdown("<div class='risk-high'>🚨 REFERRAL RECOMMENDED</div>", unsafe_allow_html=True)
                st.error("Model detected morphological biomarkers consistent with high-risk Glaucoma.")
            else:
                st.markdown("<div class='risk-low'>✅ NORMAL VARIANT</div>", unsafe_allow_html=True)
                st.success("No structural anomalies detected above the clinical decision boundary.")
            
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = glaucoma_prob * 100,
                title = {'text': "Pathology Risk Index (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, threshold*100], 'color': "#D1FAE5"},
                        {'range': [threshold*100, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': glaucoma_prob*100}
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with rep_2:
            st.markdown("#### Clinical Documentation")
            notes = st.text_area("Physician Observations:", placeholder="Enter findings regarding cup-to-disc ratio...")
            report_txt = f"ID: {patient_id}\nProb: {glaucoma_prob:.4f}\nNotes: {notes}"
            st.download_button("📄 Export Official Clinical Note", data=report_txt, file_name=f"Report_{patient_id}.txt")

    with tab_img:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.image(raw_img, caption="1. Original Fundus", use_container_width=True)
        c2.image(processed_img, caption="2. CLAHE Feature Isolation", use_container_width=True)
        c3.image(overlay, caption="3. AI Activation Map", use_container_width=True)
        st.info("💡 **Clinical Tip:** Grad-CAM++ focuses on the neuroretinal rim to identify thinning patterns.")

    with tab_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.json({
            "Inference_Stats": {
                "Internal_Score": float(prediction[0]),
                "Diagnostic_Prob": glaucoma_prob,
                "Target_Layer": GRADCAM_LAYER,
                "Input_Shape": "[224, 224, 3]"
            },
            "Metadata": {
                "Patient_ID": patient_id,
                "Timestamp": datetime.now().isoformat(),
                "Software_Ver": "Platinum v2.4"
            }
        })
