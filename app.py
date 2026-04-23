
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import pandas as pd
import os

# ==========================================
# 1. CONFIGURATION & DATABASE INIT
# ==========================================
st.set_page_config(
    page_title="OculoVision AI | Clinical Database", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "clinical_database.csv"

def save_to_database(data):
    """Saves patient screening details to a persistent CSV file."""
    df = pd.DataFrame([data])
    if not os.path.isfile(DB_FILE):
        df.to_csv(DB_FILE, index=False)
    else:
        df.to_csv(DB_FILE, mode='a', header=False, index=False)

# Ultimate Clinical CSS Theme with Pulsing Online Dot
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6;
    }
    
    /* Pulsing Online Dot */
    .online-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 600;
        color: #059669;
        margin-bottom: 20px;
    }
    .dot {
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 1);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .hero-container {
        background: radial-gradient(circle at top right, #1e3a8a, #0f172a);
        padding: 30px; border-radius: 16px; color: white; margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15); display: flex;
        justify-content: space-between; align-items: center; border: 1px solid #334155;
    }
    .hero-title { font-size: 32px; font-weight: 800; margin: 0; }
    .ehr-card { background: white; border-radius: 12px; padding: 20px; border-top: 4px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .severity-bar { height: 12px; width: 100%; background: #e2e8f0; border-radius: 10px; margin-top: 10px; position: relative; overflow: hidden;}
    .bar-fill { height: 100%; background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%); transition: width 0.8s ease-in-out; }
    .badge { padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 13px; display: inline-block; }
    .badge-critical { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
    .badge-normal { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
    </style>
""", unsafe_allow_html=True)

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"

@st.cache_resource
def load_clinical_model():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

model = load_clinical_model()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

# [apply_clahe_clinical, preprocess_for_inference, generate_gradcam logic remains same]
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
    if np.max(heatmap) == 0: return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap /= np.max(heatmap) + 1e-10
    return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="online-indicator"><span class="dot"></span> System Online</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1e3a8a; font-weight:800;'>OculoVision AI™</h2>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚙️ Calibration")
    threshold = st.slider("Clinical Decision Boundary", 0.10, 0.90, 0.60, 0.01)
    heatmap_opacity = st.slider("XAI Heatmap Blend", 0.1, 0.9, 0.5, 0.1)

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown("""
<div class="hero-container">
    <div>
        <p class="hero-title">Ophthalmic Decision Workspace</p>
        <p class="hero-subtitle">Secure AI Screening & Patient Record Integration</p>
    </div>
    <div style="text-align: right; background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 8px;">
        <span style="font-size: 12px; color: #cbd5e1; text-transform: uppercase;">Session Type</span><br>
        <span style="font-size: 16px; font-weight: 600;">EHR-Linked Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🩺 Patient Electronic Health Record (EHR)", expanded=True):
    e1, e2, e3, e4 = st.columns(4)
    patient_id = e1.text_input("Patient MRN/ID", "PT-1001")
    age = e2.number_input("Age", 1, 120, 45)
    eye = e3.selectbox("Evaluation Eye", ["Right (OD)", "Left (OS)"])
    iop = e4.number_input("IOP (mmHg)", 5, 50, 15)

uploaded_file = st.file_uploader("Drop Retinal Fundus Scan (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Inference
    raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
    model_input = prep_func(processed_img.astype(np.float32))
    
    prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    glaucoma_prob = 1.0 - float(prediction[0])
    
    # Save to Database logic
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient_ID": patient_id,
        "Age": age,
        "Eye": eye,
        "IOP": iop,
        "Glaucoma_Probability": round(glaucoma_prob, 4),
        "Diagnosis": "Glaucoma Suspect" if glaucoma_prob >= threshold else "Normal"
    }
    save_to_database(entry)
    
    # UI Results
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Results", "🔬 Neural Heatmaps", "📑 Data Log"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1.5, 1])
        with r1:
            st.markdown('<div class="ehr-card">', unsafe_allow_html=True)
            st.markdown("### AI Diagnostic Summary")
            if glaucoma_prob >= threshold:
                st.markdown("<div class='badge badge-critical'>🚨 CRITICAL FINDING: GLAUCOMA SUSPECT</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge badge-normal'>✅ ROUTINE FINDING: NORMAL LIMITS</div>", unsafe_allow_html=True)
            
            st.markdown(f"**Risk Level:** {glaucoma_prob*100:.1f}%")
            st.markdown(f'<div class="severity-bar"><div class="bar-fill" style="width: {glaucoma_prob*100}%;"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 1.0 - heatmap_opacity, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), heatmap_opacity, 0)
        st.image(overlay, caption="Grad-CAM++ Neural Activation Trace", use_container_width=True)

    with tab3:
        st.markdown("### Recent Patient Logs")
        if os.path.exists(DB_FILE):
            st.dataframe(pd.read_csv(DB_FILE).tail(10), use_container_width=True)
    
    # Patient Report Download
    report_text = f"""GLAUCOMA SCREENING REPORT\nMRN: {patient_id}\nAge: {age}\nEye: {eye}\nIOP: {iop} mmHg\nResult: {entry['Diagnosis']}\nAI Prob: {glaucoma_prob*100:.2f}%"""
    st.download_button("📥 Download Patient Report", data=report_text, file_name=f"Report_{patient_id}.txt")
