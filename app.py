import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# 1. CONFIGURATION & RADIOLOGY THEME
# ==========================================
st.set_page_config(
    page_title="OculoVision Analytics", 
    page_icon="⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast clinical CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .clinical-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .clinical-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .status-indicator {
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-high { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .status-low { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
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
# 2. PREPROCESSING & XAI LOGIC
# ==========================================
def apply_clahe_clinical(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

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
        
    enhanced = apply_clahe_clinical(masked)
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
    
    if np.max(heatmap) == 0:
        return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        
    heatmap /= np.max(heatmap) + 1e-10
    return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

# ==========================================
# 3. SIDEBAR & EHR INTEGRATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004613.png", width=70)
    st.markdown("## OculoVision Analytics")
    st.divider()
    
    st.markdown("### 📋 EHR Context")
    patient_mrn = st.text_input("Medical Record Number", "PT-88392")
    patient_age = st.number_input("Patient Age", 18, 110, 68)
    eye_side = st.radio("Ocular Laterality", ["OD (Right Eye)", "OS (Left Eye)"], horizontal=True)
    
    st.divider()
    st.markdown("### ⚙️ AI Parameters")
    threshold = st.slider("Diagnostic Sensitivity Cutoff", 0.10, 0.90, 0.60, 0.01)
    heatmap_opacity = st.slider("XAI Overlay Opacity", 0.1, 0.9, 0.5, 0.1)

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.markdown('<p class="clinical-title">Ophthalmic AI Diagnostic Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="clinical-subtitle">Automated structural evaluation of the optic nerve head using MobileNetV2.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Import Retinal Scan (DICOM/JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Inference ---
    start_time = time.time()
    with st.spinner('Processing topographic scan & generating XAI tensor...'):
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        glaucoma_prob = 1.0 - float(prediction[0])
        
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 1.0 - heatmap_opacity, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), heatmap_opacity, 0)
    
    inference_time = (time.time() - start_time) * 1000

    # --- TOP ROW: Analytics & Gauges ---
    st.markdown("### 📊 Executive Telemetry")
    met_col1, met_col2, met_col3 = st.columns([1, 1.5, 1])
    
    with met_col1:
        with st.container(border=True):
            st.metric("AI Confidence Score", f"{glaucoma_prob * 100:.1f}%", f"{((glaucoma_prob - threshold)/threshold)*100:.1f}% vs Cutoff", delta_color="inverse")
            st.metric("Inference Latency", f"{inference_time:.0f} ms")
            st.metric("Laterality", eye_side.split(" ")[0])

    with met_col2:
        with st.container(border=True):
            # Interactive Plotly Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = glaucoma_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Neuropathy Risk Index", 'font': {'size': 18}},
                delta = {'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "#dcfce7"},
                        {'range': [threshold * 100, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 4},
                        'thickness': 0.75,
                        'value': glaucoma_prob * 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with met_col3:
        with st.container(border=True):
            st.markdown("#### Clinical Disposition")
            if glaucoma_prob >= threshold:
                st.markdown('<span class="status-indicator status-high">🚨 GLAUCOMA SUSPECT</span>', unsafe_allow_html=True)
                st.markdown("High probability of structural defect. Refer for perimetry (Visual Field Test) and OCT imaging.")
            else:
                st.markdown('<span class="status-indicator status-low">✅ NORMAL LIMITS</span>', unsafe_allow_html=True)
                st.markdown("No severe neuroretinal rim thinning or C/D ratio anomalies detected. Standard follow-up.")

    # --- MIDDLE ROW: Imaging ---
    st.markdown("### 🔬 Anatomical Workspace")
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        with st.container(border=True):
            st.image(raw_img, caption="1. Input Topology", use_container_width=True)
    with img_col2:
        with st.container(border=True):
            st.image(processed_img, caption="2. CLAHE Feature Isolation", use_container_width=True)
    with img_col3:
        with st.container(border=True):
            st.image(overlay, caption="3. AI Attention Mapping (Grad-CAM++)", use_container_width=True)

    # --- BOTTOM ROW: Patient History & Export ---
    st.markdown("### 📈 Longitudinal Record")
    hist_col, export_col = st.columns([2, 1])
    
    with hist_col:
        with st.container(border=True):
            # Simulated Patient History Chart using Plotly
            dates = pd.date_range(end=datetime.now(), periods=5, freq='6M')
            iops = [15, 16, 18, 22, 24] # Simulated increasing pressure
            
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(x=dates, y=iops, mode='lines+markers', name='IOP (mmHg)', line=dict(color='#3b82f6', width=3)))
            line_fig.add_hline(y=21, line_dash="dash", line_color="red", annotation_text="Hypertension Threshold")
            line_fig.update_layout(title="Historical Intraocular Pressure (Simulated)", height=200, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="mmHg")
            st.plotly_chart(line_fig, use_container_width=True)
            
    with export_col:
        with st.container(border=True):
            st.markdown("#### Documentation")
            physician_notes = st.text_area("Findings:", placeholder="C/D ratio appears enlarged...", height=100)
            
            report = f"MRN: {patient_mrn}\nAge: {patient_age}\nEye: {eye_side}\nAI Score: {glaucoma_prob*100:.1f}%\nNotes: {physician_notes}"
            st.download_button("💾 Commit to EMR (Export)", data=report, file_name=f"EHR_{patient_mrn}.txt", use_container_width=True)
