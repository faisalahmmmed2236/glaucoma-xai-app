import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="Glaucoma AI Screening", page_icon="👁️", layout="wide")

IMAGE_SIZE = 224
GRADCAM_LAYER = "Conv_1"  # MobileNetV2 target layer

@st.cache_resource
def load_clinical_model():
    """Loads the model once and caches it in memory for speed.
    compile=False prevents deserialization TypeErrors on cloud servers."""
    return tf.keras.models.load_model('models/final_glaucoma_deploy_model.keras', compile=False)

model = load_clinical_model()
prep_func = tf.keras.applications.mobilenet_v2.preprocess_input

# ==========================================
# 2. CLINICAL PREPROCESSING
# ==========================================
def apply_clahe_clinical(img):
    """Enhances contrast for retinal vessels and optic disc."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def preprocess_for_inference(image_bytes):
    """Applies the exact masking and CLAHE used during training."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Polar Masking
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), int(min(h, w) * 0.45), 255, -1)
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # Crop to content
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, bbox_w, bbox_h = cv2.boundingRect(coords)
        masked = masked[y:y+bbox_h, x:x+bbox_w]
        
    # Enhance and Resize
    enhanced = apply_clahe_clinical(masked)
    resized = cv2.resize(enhanced, (IMAGE_SIZE, IMAGE_SIZE))
    
    return img_rgb, resized

# ==========================================
# 3. EXPLAINABLE AI (Grad-CAM++)
# ==========================================
def generate_gradcam(model, img_array, layer_name):
    """Generates diagnostic heatmaps for clinical trust."""
    img_batch = np.expand_dims(img_array, axis=0)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        # FIX: Track the inverse to highlight Glaucoma features properly
        loss = 1.0 - predictions[:, 0] 
        
    grads = tape.gradient(loss, conv_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    # Safety catch to avoid division by zero
    if np.max(heatmap) == 0:
        return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        
    heatmap /= np.max(heatmap) + 1e-10
    
    return cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

# ==========================================
# 4. FRONTEND USER INTERFACE
# ==========================================
st.title("👁️ AI-Powered Glaucoma Screening Tool")
st.markdown("""
This clinical decision-support tool utilizes a robust Deep Learning ensemble to analyze fundus imagery. 
Upload a retinal scan to receive a diagnostic probability and spatial heatmap (XAI).
""")

# Sidebar controls
with st.sidebar:
    st.header("Clinical Settings")
    # FIX: Default value set to 0.60 based on your optimum clinical threshold
    threshold = st.slider("Diagnostic Threshold", min_value=0.10, max_value=0.90, value=0.60, step=0.01, 
                          help="Adjust based on Youden's J statistic from validation.")
    st.info("Operating Model: **MobileNetV2 (Ensemble Best Fold)**")

uploaded_file = st.file_uploader("Upload Fundus Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    
    with st.spinner('Processing retinal image...'):
        # Process Image
        raw_img, processed_img = preprocess_for_inference(uploaded_file.read())
        model_input = prep_func(processed_img.astype(np.float32))
        
        # Inference
        prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
        
        # FIX: Extract the probability and INVERT it so it correctly maps to Glaucoma
        normal_prob = float(prediction[0])
        glaucoma_prob = 1.0 - normal_prob
        
        # Generate XAI
        heatmap = generate_gradcam(model, model_input, GRADCAM_LAYER)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed_img, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)

    # Display Results
    with col1:
        st.subheader("1. Original Image")
        st.image(raw_img, use_container_width=True)
        
    with col2:
        st.subheader("2. Preprocessed")
        st.image(processed_img, caption="CLAHE + Polar Masking", use_container_width=True)
        
    with col3:
        st.subheader("3. XAI Activation")
        st.image(overlay, caption="Grad-CAM Focus Area", use_container_width=True)

    st.divider()
    
    # Clinical Report
    st.subheader("📊 Diagnostic Output")
    if glaucoma_prob >= threshold:
        st.error("🚨 **System Flag: Referral Recommended (Glaucoma Suspect)**")
    else:
        st.success("✅ **System Flag: Normal (Routine Follow-up)**")
        
    st.metric(label="Glaucoma Probability", value=f"{glaucoma_prob * 100:.2f}%")