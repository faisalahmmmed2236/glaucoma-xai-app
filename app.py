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
DB_FILE = "patient_history_log.csv"

# Professional Dark-Mode Clinical CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #020617; color: #f8fafc; }
    .stApp { background-color: #020617; }
    
    /* Sidebar Clinical Cards */
    .sidebar-card {
        background-color: #1e3a8a; color: white !important; padding: 15px; 
        border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #60a5fa;
    }
    
    /* Pulsing Online Indicator */
    .online-indicator { display: flex; align-items: center; gap: 8px; color: #10b981; font-weight: 700; margin-bottom: 15px; }
    .pulse { height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; animation: pulse-anim 2s infinite; }
    @keyframes pulse-anim { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATOR WITH MEDICAL LOGIC
# ==========================================
def create_medical_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(200, 20, "OFFICIAL GLAUCOMA SCREENING REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient ID: {data['Patient_ID']}")
    pdf.cell(100, 10, f"Date: {data['Timestamp']}", ln=True)
    pdf.cell(100, 10, f"Age: {data['Age']}")
    pdf.cell(100, 10, f"Eye: {data['Eye']}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    status_color = (220, 38, 38) if data['Result'] == "GLAUCOMA SUSPECT" else (22, 163, 74)
    pdf.set_text_color(*status_color)
    pdf.cell(200, 10, f"DIAGNOSTIC STATUS: {data['Result']}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(50, 50, 50)
    
    # Automated Clinical Explanation
    if data['Result'] == "GLAUCOMA SUSPECT":
        explanation = ("The AI detected thinning of the neuroretinal rim and enlargement of the optic cup. "
                       "This pattern suggests increased intraocular pressure damage to the optic nerve fibers, "
                       "a primary indicator of glaucoma.")
    else:
        explanation = ("The optic disc morphology shows a healthy neuroretinal rim with no significant "
                       "excavation or 'cupping.' Structural patterns align with healthy ocular physiology.")
    
    pdf.multi_cell(0, 10, f"Clinical Insight: {explanation}")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. CORE AI & DB LOGIC
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

# [Insert preprocess_for_inference and generate_gradcam functions here]
# (Keep your existing computer vision functions)

# ==========================================
# 4. SIDEBAR (HISTORY & AUTO-UPDATE)
# ==========================================
with st.sidebar:
    st.markdown('<div class="online-indicator"><span class="pulse"></span> SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#60a5fa;'>OculoVision PRO</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="sidebar-card">
            <b>🔬 XAI Powered</b>
            Biomarker visualization enabled.
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 New Patient (Reset ID)"):
        st.session_state.p_count = st.session_state.get('p_count', 1000) + 1
        st.session_state.processed = False
        st.rerun()

    st.divider()
    st.markdown("### 📁 Global Patient History")
    if os.path.exists(DB_FILE):
        history_df = pd.read_csv(DB_FILE)
        st.dataframe(history_df[['Patient_ID', 'Result']].tail(10), use_container_width=True)

# ==========================================
# 5. MAIN WORKSPACE
# ==========================================
if 'p_count' not in st.session_state: st.session_state.p_count = 1001

st.title("👁️ Diagnostic Command Center")

with st.container():
    c1, c2, c3 = st.columns(3)
    p_id = c1.text_input("Patient ID", value=f"PX-{st.session_state.p_count}")
    age = c2.number_input("Age", 1, 110, 60)
    eye = c3.selectbox("Target Eye", ["Right (OD)", "Left (OS)"])

uploaded_file = st.file_uploader("Upload Fundus Scan", type=["jpg", "png"])

if uploaded_file:
    # Inference Math
    # [Run your engine.predict and processing here]
    # For this example, let's assume we have 'glaucoma_prob' and 'diag_result'
    glaucoma_prob = 0.85 # Example
    diag_result = "GLAUCOMA SUSPECT" if glaucoma_prob >= 0.60 else "NORMAL"

    # Auto-Save Entry
    db_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Patient_ID": p_id, "Age": age, "Eye": eye,
        "Prob": f"{glaucoma_prob*100:.1f}%", "Result": diag_result
    }
    
    if not st.session_state.get('processed', False):
        save_to_history(db_data)
        st.session_state.processed = True

    # PDF Download
    pdf_bytes = create_medical_pdf(db_data)
    st.download_button("📥 Download Official PDF Report", data=pdf_bytes, file_name=f"Report_{p_id}.pdf")

    st.success(f"Analysis Complete for {p_id}. Record logged in sidebar history.")
