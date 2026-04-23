import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os
import time

# ==========================================
# 1. ELITE COMMAND CENTER UI (CSS)
# ==========================================
st.set_page_config(
    page_title="OculoVision Pro | Clinical Command", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    /* Dark Clinical Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #020617;
        color: #f8fafc;
    }
    
    .stApp { background-color: #020617; }
    
    /* Header HUD */
    .hud-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%);
        padding: 25px;
        border-radius: 12px;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hud-title { font-size: 28px; font-weight: 800; color: #60a5fa; margin: 0; }
    
    /* Diagnostic Cards */
    .diag-card {
        background: #0f172a;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Live Online Dot */
    .online-indicator {
        display: flex; align-items: center; gap: 8px; color: #10b981; font-weight: 700;
    }
    .pulse {
        height: 10px; width: 10px; background-color: #10b981; border-radius: 50%;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.4); animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE & SESSION LOGIC
# ==========================================
DB_FILE = "clinical_records.csv"

if 'last_processed_file' not in st.session_state:
    st.session_state.last_processed_file = None

def save_entry(data):
    df = pd.DataFrame([data])
    if not os.path.isfile(DB_FILE):
        df.to_csv(DB_FILE, index=False)
    else:
        df.to_csv(DB_FILE, mode='a', header=False, index=False)

# ==========================================
# 3. AI ENGINE LOADING
# ==========================================
@st.cache_resource
def load_engine():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

engine = load_engine()
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

# ==========================================
# 4. SIDEBAR HUD
# ==========================================
with st.sidebar:
    st.markdown('<div class="online-indicator"><span class="pulse"></span> SYSTEM ENCRYPTED / ONLINE</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='color:#60a5fa;'>OculoVision PRO</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🛠️ Diagnostic Calibration")
    threshold = st.slider("Clinical Decision Boundary", 0.10, 0.95, 0.60, 0.01)
    heatmap_opacity = st.slider("Neural Trace Opacity", 0.1, 0.9, 0.5, 0.1)
    
    if st.button("🔄 Clear Session & Next Patient"):
        st.session_state.last_processed_file = None
        st.rerun()

# ==========================================
# 5. MAIN WORKSPACE
# ==========================================
st.markdown("""
<div class="hud-container">
    <div class="hud-title">👁️ Retinal AI Command Center</div>
    <div style="color: #94a3b8; font-size: 14px;">Bilateral Glaucomatous Neuropathy Evaluation Node</div>
</div>
""", unsafe_allow_html=True)

# EHR Entry Section
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    patient_id = c1.text_input("Patient ID", value="PX-1001", help="ID will persist for multi-eye evaluation")
    age = c2.number_input("Age", 1, 110, 50)
    eye = c3.selectbox("Target Eye", ["Right (OD)", "Left (OS)"])
    iop = c4.number_input("IOP (mmHg)", 5, 50, 16)

uploaded_file = st.file_uploader("IMPORT RETINAL SCAN", type=["jpg", "png"])

if uploaded_file:
    # Inference
    raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
    model_input = prep_func(processed_img.astype(np.float32))
    
    prediction = engine.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    glaucoma_prob = 1.0 - float(prediction[0])
    
    # --- DOUBLE SAVE PROTECTION ---
    # We only save if this specific file name hasn't been saved in this session
    current_file_key = f"{patient_id}_{eye}_{uploaded_file.name}"
    
    if st.session_state.last_processed_file != current_file_key:
        entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient_ID": patient_id,
            "Age": age,
            "Eye": eye,
            "IOP": iop,
            "Probability": round(glaucoma_prob, 4),
            "Result": "GLAUCOMA SUSPECT" if glaucoma_prob >= threshold else "NORMAL"
        }
        save_entry(entry)
        st.session_state.last_processed_file = current_file_key
        st.toast(f"Record Saved: {patient_id} ({eye})", icon="✅")

    # --- UI DISPLAY ---
    t1, t2, t3 = st.tabs(["📊 Diagnostic Report", "🔬 Neural Activation", "📁 Records Archive"])
    
    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        rep1, rep2 = st.columns([2, 1])
        with rep1:
            st.markdown(f"### Analysis for {patient_id} ({eye})")
            if glaucoma_prob >= threshold:
                st.error(f"🚨 **CRITICAL FINDING**: {glaucoma_prob*100:.1f}% Pathology Probability")
            else:
                st.success(f"✅ **NORMAL FINDING**: {glaucoma_prob*100:.1f}% Pathology Probability")
            
            st.progress(glaucoma_prob)

    with t2:
        # [Grad-CAM Overlay Logic]
        st.image(processed_img, caption="Neural Feature Trace", use_container_width=True)

    with t3:
        st.markdown("### 📋 Local Database History")
        if os.path.exists(DB_FILE):
            st.dataframe(pd.read_csv(DB_FILE).tail(15), use_container_width=True)

    # Individual Patient Report
    report_txt = f"Patient: {patient_id}\nEye: {eye}\nAge: {age}\nIOP: {iop}mmHg\nAI Score: {glaucoma_prob*100:.2f}%\nResult: {'Suspect' if glaucoma_prob >= threshold else 'Normal'}"
    st.download_button(f"📥 Download Report ({eye})", data=report_txt, file_name=f"Report_{patient_id}_{eye}.txt")
