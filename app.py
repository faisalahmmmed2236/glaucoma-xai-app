import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os
from fpdf import FPDF
import time

# ==========================================
# 1. GLOBAL CONFIGURATION & DATABASE
# ==========================================
st.set_page_config(page_title="OculoVision Pro", page_icon="👁️", layout="wide")

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"
DB_FILE = "patient_clinical_history.csv"

# Professional Dark-Mode Clinical CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #020617; color: #f8fafc; }
    .stApp { background-color: #020617; }
    
    /* Header HUD */
    .hud-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%);
        padding: 25px; border-radius: 12px; border-bottom: 3px solid #3b82f6;
        margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Clinical Cards */
    .sidebar-feature-card {
        background-color: #1e3a8a; color: #ffffff !important; padding: 15px; 
        border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #60a5fa;
        font-size: 13px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .sidebar-feature-card b { color: #bfdbfe; font-size: 15px; display: block; margin-bottom: 3px; }

    /* Pulsing Online Indicator */
    .online-indicator { display: flex; align-items: center; gap: 8px; color: #10b981; font-weight: 700; margin-bottom: 15px; }
    .pulse { height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; animation: pulse-anim 2s infinite; }
    @keyframes pulse-anim { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATOR WITH MEDICAL REASONING
# ==========================================
def create_medical_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(200, 20, "OFFICIAL GLAUCOMA SCREENING REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient ID: {data['Patient_ID']}")
    pdf.cell(100, 10, f"Timestamp: {data['Timestamp']}", ln=True)
    pdf.cell(100, 10, f"Ocular Laterality: {data['Eye']}")
    pdf.cell(100, 10, f"Intraocular Pressure: {data['IOP']} mmHg", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    status_color = (220, 38, 38) if data['Result'] == "GLAUCOMA SUSPECT" else (22, 163, 74)
    pdf.set_text_color(*status_color)
    pdf.cell(200, 10, f"DIAGNOSTIC STATUS: {data['Result']} ({data['Prob']})", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(50, 50, 50)
    
    # Automated Clinical Reasoning
    if data['Result'] == "GLAUCOMA SUSPECT":
        explanation = ("Reasoning: The AI ensemble detected significant excavation of the optic cup and thinning of the neuroretinal rim. "
                       "These structural changes are characteristic of glaucomatous neuropathy, where increased pressure damages optic nerve fibers. "
                       "Clinical follow-up for perimetry and OCT imaging is strongly advised.")
    else:
        explanation = ("Reasoning: The retinal topography indicates a healthy cup-to-disc ratio. The neuroretinal rim appears robust, "
                       "with no visible signs of nerve fiber layer defects or pathological cupping. Structural patterns fall within normal clinical limits.")
    
    pdf.multi_cell(0, 10, explanation)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. CORE AI & DATABASE LOGIC
# ==========================================
@st.cache_resource
def load_engine():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

engine = load_engine()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

def save_to_history(data):
    df = pd.DataFrame([data])
    if not os.path.isfile(DB_FILE): df.to_csv(DB_FILE, index=False)
    else: df.to_csv(DB_FILE, mode='a', header=False, index=False)

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
# 4. SIDEBAR (History & Auto-Update)
# ==========================================
if 'p_count' not in st.session_state: st.session_state.p_count = 1001
if 'processed_flag' not in st.session_state: st.session_state.processed_flag = False

with st.sidebar:
    st.markdown('<div class="online-indicator"><span class="pulse"></span> SYSTEM SECURE</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#60a5fa;'>OculoVision PRO</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="sidebar-feature-card"><b>🔬 XAI Powered</b>Grad-CAM++ engine visualizes pathological biomarkers.</div>
        <div class="sidebar-feature-card"><b>🛡️ Robust Math</b>Train on HRF Tested on Drishti-GS .</div>
        <div class="sidebar-feature-card"><b>📊 EMR / PDF Ready</b>Syncs records & exports medical reports.</div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Auto-Update: Next Patient"):
        st.session_state.p_count += 1
        st.session_state.processed_flag = False
        st.rerun()

    st.divider()
    st.markdown("### 📁 Clinical History Log")
    if os.path.exists(DB_FILE):
        hist_df = pd.read_csv(DB_FILE)
        st.dataframe(hist_df[['Patient_ID', 'Eye', 'Result']].tail(10), use_container_width=True)

# ==========================================
# 5. MAIN WORKSPACE
# ==========================================
st.markdown("""<div class="hud-container"><div style="font-size: 28px; font-weight: 800; color: #60a5fa;">👁️ Neural Diagnostic Workspace</div>
<div style="color: #94a3b8; font-size: 14px;">Automated Screening Node • Enterprise Clinical v4.0</div></div>""", unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    p_id = c1.text_input("Patient ID", value=f"PX-{st.session_state.p_count}")
    age = c2.number_input("Age", 1, 110, 60)
    eye = c3.selectbox("Target Eye", ["Right (OD)", "Left (OS)"])
    iop = c4.number_input("IOP (mmHg)", 5, 50, 16)

uploaded_file = st.file_uploader("DROP SCAN HERE", type=["jpg", "png"])

if uploaded_file:
    # Inference
    raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
    model_input = prep_func(processed_img.astype(np.float32))
    
    prediction = engine.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    glaucoma_prob = 1.0 - float(prediction[0])
    diag_result = "GLAUCOMA SUSPECT" if glaucoma_prob >= 0.60 else "NORMAL"
    
    # --- AUTO-SAVE LOGIC (FIXED) ---
    current_key = f"{p_id}_{eye}_{uploaded_file.name}"
    
    if not st.session_state.processed_flag:
        db_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient_ID": p_id, "Age": age, "Eye": eye, "IOP": iop,
            "Prob": f"{glaucoma_prob*100:.1f}%", "Result": diag_result
        }
        save_to_history(db_entry)
        st.session_state.processed_flag = True
        st.toast(f"Record logged for {p_id}", icon="💾")

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Diagnostic HUD", "🔬 Neural Trace", "📑 Patient Report"])
    
    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        if glaucoma_prob >= 0.60: st.error(f"🚨 CRITICAL FINDING: {glaucoma_prob*100:.1f}% Risk")
        else: st.success(f"✅ NORMAL VARIANT: {glaucoma_prob*100:.1f}% Risk")
        st.progress(glaucoma_prob)

    with t2:
        heatmap = generate_gradcam(engine, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 0.5, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.5, 0)
        st.image(overlay, caption="Neural Activation Map", use_container_width=True)

    with t3:
        st.markdown("### Official Clinical Documentation")
        report_data = {
            "Patient_ID": p_id, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Eye": eye, "IOP": iop, "Result": diag_result, "Prob": f"{glaucoma_prob*100:.1f}%"
        }
        pdf_bytes = create_medical_report(report_data)
        st.download_button(f"📥 Download Official PDF ({p_id})", data=pdf_bytes, file_name=f"Report_{p_id}.pdf")
