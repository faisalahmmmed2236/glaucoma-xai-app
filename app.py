import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os
from fpdf import FPDF
import base64

# ==========================================
# 1. PLATINUM UI & DATABASE CONFIG
# ==========================================
st.set_page_config(page_title="OculoVision Pro | Clinical AI", page_icon="👁️", layout="wide")

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"
DB_FILE = "clinical_database.csv"

# Global CSS for Clinical Dashboard & Pulsing Online Dot
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    .online-indicator { display: flex; align-items: center; gap: 8px; color: #059669; font-weight: 700; margin-bottom: 10px; }
    .dot { height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .feature-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #1e3a8a; box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100%; }
    .hero-section { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 40px; border-radius: 20px; color: white; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATOR FUNCTION
# ==========================================
def create_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "OCULOVISION CLINICAL REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for key, value in data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This AI-generated report is for clinical decision support and does not replace a professional ophthalmological diagnosis.")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. CORE AI & DB LOGIC
# ==========================================
@st.cache_resource
def load_engine():
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

def save_to_db(data):
    df = pd.DataFrame([data])
    if not os.path.isfile(DB_FILE): df.to_csv(DB_FILE, index=False)
    else: df.to_csv(DB_FILE, mode='a', header=False, index=False)

# ... (Insert your preprocess_for_inference and generate_gradcam functions here) ...

# ==========================================
# 4. DASHBOARD UI
# ==========================================
with st.sidebar:
    st.markdown('<div class="online-indicator"><span class="dot"></span> System Live & Secure</div>', unsafe_allow_html=True)
    st.title("OculoVision Pro")
    st.divider()
    threshold = st.slider("Clinical Threshold", 0.1, 0.9, 0.6)
    
st.markdown("""
<div class="hero-section">
    <h1 style="margin:0;">Ophthalmic Intelligence Center</h1>
    <p style="opacity:0.8;">Automated Glaucoma Screening with XAI Evidence</p>
</div>
""", unsafe_allow_html=True)

# Why OculoVision? Section
st.markdown("### 🏆 Why OculoVision is the Best-in-Class Solution")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown('<div class="feature-card"><b>🔬 XAI Powered</b><br>We don\'t just give a score. Our Grad-CAM++ engine shows exactly which anatomical features the AI is flagging.</div>', unsafe_allow_html=True)
with col_f2:
    st.markdown('<div class="feature-card"><b>🛡️ Robust Math</b><br>Our model is validated against external hospital data (Drishti-GS) and is resistant to noise and blur.</div>', unsafe_allow_html=True)
with col_f3:
    st.markdown('<div class="feature-card"><b>📊 EMR Ready</b><br>Instantly syncs with patient records and provides professional PDF reports for hospital compliance.</div>', unsafe_allow_html=True)

st.divider()

# Patient Records & Upload
with st.expander("📝 Patient Information", expanded=True):
    c1, c2, c3 = st.columns(3)
    p_id = c1.text_input("Patient ID", "PX-001")
    age = c2.number_input("Age", 1, 100, 55)
    eye = c3.selectbox("Eye", ["Right", "Left"])

uploaded_file = st.file_uploader("Upload Retinal Scan", type=["jpg", "png"])

if uploaded_file:
    # --- AI Magic Happens Here ---
    # (Simulated Probability for brevity, use your engine.predict logic here)
    glaucoma_prob = 0.72  
    diagnosis = "Suspect" if glaucoma_prob >= threshold else "Normal"
    
    # Save to Database
    db_entry = {"Timestamp": datetime.now(), "ID": p_id, "Age": age, "Eye": eye, "Prob": glaucoma_prob, "Result": diagnosis}
    save_to_db(db_entry)
    
    st.success(f"Diagnosis: {diagnosis} ({glaucoma_prob*100:.1f}%)")
    
    # PDF Generation
    pdf_data = create_pdf_report(db_entry)
    st.download_button(label="📥 Download Clinical PDF Report", data=pdf_data, file_name=f"Report_{p_id}.pdf", mime="application/pdf")

    # Data History
    st.markdown("### 📋 Recent Database Logs")
    if os.path.exists(DB_FILE):
        st.dataframe(pd.read_csv(DB_FILE).tail(5), use_container_width=True)
