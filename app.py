import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. CONFIGURATION & CACHING
# ==========================================
st.set_page_config(
    page_title="Glaucoma AI Screening", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, clinical look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .report-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"  # MobileNetV2 target layer

@st.cache_resource
def load_clinical_model():
    """Loads the model once and caches it in memory for speed."""
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

model = load_clinical_model()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

# ==========================================
# 2. CLINICAL PREPROCESSING
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

# ==========================================
# 3. EXPLAINABLE AI (Grad-CAM++)
# ==========================================
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
# 4. FRONTEND USER INTERFACE
# ==========================================

# --- Header Section ---
st.markdown('<p class="main-header">👁️ AI-Powered Glaucoma Screening</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Clinical Decision-Support Tool via Deep Learning Ensemble</p>', unsafe_allow_html=True)

with st.expander("📖 How to use this tool"):
    st.write("""
    1. **Upload** a clear, centered fundus retinal image (JPG or PNG).
    2. The system will automatically apply **Polar Masking** and **CLAHE** contrast enhancement.
    3. The model will evaluate the image and output a diagnostic probability.
    4. Review the **XAI Heatmap** to see which anatomical structures drove the AI's decision.
    """)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3203/3203859.png", width=80) # Placeholder medical icon
    st.header("⚙️ Clinical Settings")
    threshold = st.slider("Diagnostic Threshold", min_value=0.10, max_value=0.90, value=0.60, step=0.01, 
                          help="Adjust based on Youden's J statistic from external validation.")
    
    st.markdown("---")
    st.info("**Model Architecture:**\nMobileNetV2 (Ensemble Best Fold)\n\n**External AUC:** 0.673")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for research and demonstration purposes only and does not replace professional medical diagnosis.")

# --- Main App ---
uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"], help="Ensure the optic disc is clearly visible.")

if uploaded_file is not None:
    st.markdown("---")
    
    with st.spinner('⚙️ Analyzing retinal structures...'):
        # Inference & Math
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        normal_prob = float(prediction[0])
        glaucoma_prob = 1.0 - normal_prob
        
        # XAI
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)

    # --- Visual Outputs ---
    st.markdown("### 🔬 Image Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(raw_img, caption="1. Original Image", use_container_width=True)
    with col2:
        st.image(processed_img, caption="2. Preprocessed (CLAHE)", use_container_width=True)
    with col3:
        st.image(overlay, caption="3. XAI Activation Map", use_container_width=True)

    # --- Final Report ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Clinical Diagnostic Report")
    
    with st.container():
        rep_col1, rep_col2 = st.columns([2, 1])
        
        with rep_col1:
            if glaucoma_prob >= threshold:
                st.error("🚨 **System Flag: Referral Recommended (Glaucoma Suspect)**")
                st.write("The model has detected structural anomalies consistent with Glaucoma. Please review the XAI map for clinical correlation.")
            else:
                st.success("✅ **System Flag: Normal (Routine Follow-up)**")
                st.write("No significant glaucomatous patterns detected. Standard routine follow-up recommended.")
                
            # Visual Confidence Bar
            st.write("**AI Confidence Level:**")
            st.progress(glaucoma_prob)

        with rep_col2:
            st.metric(
                label="Glaucoma Probability", 
                value=f"{glaucoma_prob * 100:.1f}%", 
                delta=f"{(glaucoma_prob - threshold) * 100:.1f}% from threshold",
                delta_color="inverse" if glaucoma_prob >= threshold else "normal"
            )
