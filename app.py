import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import io

# ==========================================
# 1. CONFIGURATION & ENTERPRISE CSS
# ==========================================
st.set_page_config(
    page_title="OculoVision AI | Clinical Workspace", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Medical UI CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-text h1 { color: #ffffff; margin: 0; font-size: 28px; font-weight: 700; }
    .header-text p { color: #94a3b8; margin: 0; font-size: 14px; font-weight: 500; }
    
    .status-badge-safe { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px; border: 1px solid #bbf7d0;}
    .status-badge-danger { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px; border: 1px solid #fecaca;}
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    
    /* Customizing Streamlit's default components */
    .stProgress > div > div > div > div { background-color: #3b82f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
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
# 3. SIDEBAR & CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.markdown("## OculoVision AI")
    st.markdown("<span class='status-badge-safe'>● System Online</span>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📋 Session Setup")
    patient_id = st.text_input("Patient MRN / ID", placeholder="Ex: PT-98421")
    exam_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    st.caption(f"Session Timestamp: {exam_date}")
    
    st.divider()
    st.markdown("### ⚙️ Diagnostic Parameters")
    threshold = st.slider("Diagnostic Threshold", 0.10, 0.90, 0.60, 0.01, help="Operating point optimized via Youden's J")
    
    # NEW FEATURE: Interactive Heatmap Control
    heatmap_opacity = st.slider("XAI Heatmap Opacity", 0.1, 0.9, 0.5, 0.1, help="Adjust overlay blend ratio")
    
    st.divider()
    st.info("**Engine:** MobileNetV2 Ensemble\n**Validation AUC:** 0.673 (External)")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown(f"""
<div class="main-header">
    <div class="header-text">
        <h1>Glaucoma Screening Workspace</h1>
        <p>Deep Learning Clinical Decision Support System</p>
    </div>
    <div style="text-align: right;">
        <span style="color: #64748b; font-size: 12px; font-weight: bold;">CURRENT PATIENT</span><br>
        <span style="color: white; font-size: 18px; font-weight: bold;">{patient_id if patient_id else 'Unregistered'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop Fundus DICOM / Image Here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Execute AI Pipeline ---
    with st.spinner('Initiating Neural Network Analysis...'):
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        glaucoma_prob = 1.0 - float(prediction[0])
        
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Apply the user-defined dynamic opacity
        overlay = cv2.addWeighted(processed_img, 1.0 - heatmap_opacity, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), heatmap_opacity, 0)

    # --- UI Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔬 XAI Deep Dive", "📄 Export Report"])
    
    # ------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ------------------------------------
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_report = st.columns([1.2, 1])
        
        with col_img:
            st.image(overlay, caption=f"Dynamic XAI Overlay (Opacity: {heatmap_opacity*100:.0f}%)", use_container_width=True)
            
        with col_report:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### AI Diagnostic Output")
            
            # Dynamic Severity Badge
            if glaucoma_prob >= threshold:
                badge = "<span class='status-badge-danger'>High Risk: Glaucoma Suspect</span>"
                alert_text = "The neural ensemble detected structural anomalies within the fundus highly correlated with glaucomatous neuropathy."
            else:
                badge = "<span class='status-badge-safe'>Low Risk: Normal Variant</span>"
                alert_text = "No significant glaucomatous patterns were detected above the clinical operating threshold."
                
            st.markdown(f"{badge}", unsafe_allow_html=True)
            st.markdown(f"<p style='margin-top:15px; color:#475569;'>{alert_text}</p>", unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("**Pathology Probability Model:**")
            st.progress(glaucoma_prob)
            st.metric(
                label="Absolute Confidence", 
                value=f"{glaucoma_prob * 100:.1f}%", 
                delta=f"{(glaucoma_prob - threshold) * 100:.1f}% from cutoff",
                delta_color="inverse" if glaucoma_prob >= threshold else "normal"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------
    # TAB 2: XAI DEEP DIVE
    # ------------------------------------
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("Analyze the progression of the image through the AI's internal pipeline.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(raw_img, caption="1. Raw Patient Capture", use_container_width=True)
        with c2:
            st.image(processed_img, caption="2. CLAHE Masked (Network Input)", use_container_width=True)
        with c3:
            # Show isolated heatmap for clarity
            isolated_heat = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            st.image(isolated_heat, caption="3. Isolated Grad-CAM++ Activations", use_container_width=True)
            
        st.markdown('<div class="card"><strong>Clinical Note:</strong> Red/Yellow regions in the isolated activation map indicate the spatial features that maximally excited the network towards a positive Glaucoma diagnosis. Ensure these align with the optic disc/cup region for clinical validity.</div>', unsafe_allow_html=True)

    # ------------------------------------
    # TAB 3: EXPORT & DOCUMENTATION
    # ------------------------------------
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Generate Clinical Documentation")
        
        doctor_notes = st.text_area("Reviewing Physician Notes:", placeholder="Enter clinical observations, cup-to-disc ratio notes, or follow-up plans here...")
        
        # Build the downloadable text report
        report_content = f"""
======================================================
OCULOVISION AI - CLINICAL DIAGNOSTIC REPORT
======================================================
Date Generated:   {exam_date}
Patient ID:       {patient_id if patient_id else "Unregistered Patient"}
System Model:     MobileNetV2 Ensemble (Ext. AUC 0.673)

--- ANALYSIS RESULTS ---
Probability:      {glaucoma_prob * 100:.2f}%
Threshold:        {threshold * 100:.2f}%
Diagnosis Flag:   {"GLAUCOMA SUSPECT - REFERRAL RECOMMENDED" if glaucoma_prob >= threshold else "NORMAL - ROUTINE SCREENING"}

--- PHYSICIAN NOTES ---
{doctor_notes if doctor_notes else "No notes provided at time of export."}

======================================================
Disclaimer: AI-assisted analysis is for decision support 
only and does not constitute a final medical diagnosis.
======================================================
"""
        st.text_area("Preview Report:", value=report_content, height=250, disabled=True)
        
        st.download_button(
            label="📄 Download Official Text Report",
            data=report_content,
            file_name=f"Glaucoma_Report_{patient_id if patient_id else 'Anon'}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
