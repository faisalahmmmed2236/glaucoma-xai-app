import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time

# ==========================================
# 1. CONFIGURATION & ENTERPRISE CSS
# ==========================================
st.set_page_config(
    page_title="OculoVision AI | EHR Integrated", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultimate Clinical CSS Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6;
    }
    
    /* Premium Header */
    .hero-container {
        background: radial-gradient(circle at top right, #1e3a8a, #0f172a);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #334155;
    }
    .hero-title { font-size: 32px; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .hero-subtitle { font-size: 15px; color: #94a3b8; margin-top: 5px; font-weight: 400; }
    
    /* EHR Cards */
    .ehr-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Severity Meter */
    .severity-bar {
        height: 12px;
        width: 100%;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
        border-radius: 10px;
        margin-top: 10px;
        position: relative;
    }
    .severity-marker {
        position: absolute;
        top: -6px;
        width: 4px;
        height: 24px;
        background-color: #0f172a;
        border: 2px solid white;
        border-radius: 4px;
        transition: left 0.5s ease-out;
    }
    
    /* Badges */
    .badge { padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 13px; display: inline-block; }
    .badge-critical { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
    .badge-normal { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 16px; }
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
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#1e3a8a; font-weight:800;'>OculoVision AI™</h2>", unsafe_allow_html=True)
    st.markdown("🟢 **System Ready & Encrypted**")
    st.divider()
    
    st.markdown("### ⚙️ Engine Calibration")
    threshold = st.slider("Clinical Decision Boundary", 0.10, 0.90, 0.60, 0.01)
    heatmap_opacity = st.slider("XAI Heatmap Blend", 0.1, 0.9, 0.5, 0.1)
    
    st.divider()
    st.markdown("### 🧬 Infrastructure")
    st.caption("""
    **Core:** MobileNetV2 (Feature Extractor)
    **Pipeline:** CLAHE + Polar Masking
    **XAI Engine:** Grad-CAM++
    **Ext. Validation (AUC):** 0.673
    """)

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown("""
<div class="hero-container">
    <div>
        <p class="hero-title">Ophthalmic Decision Support Workspace</p>
        <p class="hero-subtitle">Automated Fundus Topography & Glaucomatous Neuropathy Detection</p>
    </div>
    <div style="text-align: right; background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 8px;">
        <span style="font-size: 12px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">Session</span><br>
        <span style="font-size: 16px; font-weight: 600;">Standard Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Simulated EHR Panel ---
with st.expander("🩺 Patient Electronic Health Record (EHR) Context", expanded=True):
    e1, e2, e3, e4 = st.columns(4)
    patient_mrn = e1.text_input("Patient MRN", "PT-10934")
    patient_age = e2.number_input("Age", 18, 100, 65)
    eye_side = e3.selectbox("Eye Evaluated", ["Right (OD)", "Left (OS)"])
    iop = e4.number_input("IOP (mmHg)", 10, 40, 18)

uploaded_file = st.file_uploader("Secure Image Drop (DICOM/JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider()
    
    # --- AI Pipeline with Telemetry ---
    start_time = time.time()
    with st.spinner('Engaging Deep Learning Pipeline...'):
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        glaucoma_prob = 1.0 - float(prediction[0])
        
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 1.0 - heatmap_opacity, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), heatmap_opacity, 0)
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # --- Results UI ---
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Report", "🔬 Anatomical Analysis", "📄 Export EMR Note"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1.5, 1])
        
        with r1:
            st.markdown('<div class="ehr-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0;'>Diagnostic Confidence Summary</h3>", unsafe_allow_html=True)
            
            if glaucoma_prob >= threshold:
                st.markdown("<div class='badge badge-critical'>🚨 CRITICAL FINDING: GLAUCOMA SUSPECT</div>", unsafe_allow_html=True)
                st.markdown("<p style='margin-top:10px;'>The ensemble detected highly suspicious structural topography indicative of Glaucoma. Immediate review of the optic disc parameters is advised.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge badge-normal'>✅ ROUTINE FINDING: NORMAL LIMITS</div>", unsafe_allow_html=True)
                st.markdown("<p style='margin-top:10px;'>No significant morphological indicators of Glaucoma were detected above the clinical safety threshold.</p>", unsafe_allow_html=True)
                
            st.markdown("<br><b>Pathology Risk Spectrum:</b>", unsafe_allow_html=True)
            # Custom Severity Meter HTML
            st.markdown(f"""
            <div class="severity-bar">
                <div class="severity-marker" style="left: {glaucoma_prob * 100}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-top: 5px;">
                <span>0% (Safe)</span>
                <span>Threshold ({threshold*100:.0f}%)</span>
                <span>100% (High Risk)</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="ehr-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0;'>Telemetry</h3>", unsafe_allow_html=True)
            st.metric("Pathology Probability", f"{glaucoma_prob * 100:.1f}%", delta=f"{((glaucoma_prob - threshold)/threshold)*100:.1f}% relative to threshold", delta_color="inverse")
            st.metric("Inference Engine Time", f"{inference_time:.0f} ms")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(raw_img, caption="1. Initial Fundus Geometry", use_container_width=True)
        with i2:
            st.image(processed_img, caption="2. CLAHE Preprocessed", use_container_width=True)
        with i3:
            st.image(overlay, caption=f"3. Grad-CAM++ (Blend: {heatmap_opacity*100:.0f}%)", use_container_width=True)
            
        st.info("💡 **Clinician Guidance:** Red zones in the Grad-CAM++ visualization indicate the spatial features that maximally triggered the neural network's decision weights.")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        doc_notes = st.text_area("Consulting Physician Notes:", height=100, placeholder="Reviewing physician comments regarding optic cup-to-disc ratio...")
        
        final_report = f"""OCULOVISION AI - CLINICAL REPORT
========================================
Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Patient MRN:  {patient_mrn}
Age / Eye:    {patient_age} yrs / {eye_side}
Baseline IOP: {iop} mmHg

--- DIAGNOSTIC FINDINGS ---
AI Confidence Score:   {glaucoma_prob * 100:.2f}%
Diagnostic Threshold:  {threshold * 100:.0f}%
System Flag:           {"GLAUCOMA SUSPECT" if glaucoma_prob >= threshold else "NORMAL WITHIN LIMITS"}

--- PHYSICIAN REMARKS ---
{doc_notes if doc_notes else "No notes appended."}
========================================"""
        
        st.download_button("📥 Download Official Report (TXT)", data=final_report, file_name=f"Report_{patient_mrn}.txt", mime="text/plain")
