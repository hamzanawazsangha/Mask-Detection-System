import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
from huggingface_hub import hf_hub_download

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Custom CSS for UI
st.markdown("""
<style>
    .header {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(45deg, #1E88E5, #0D47A1);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        background: #f8f9fa;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .confidence-bar {
        height: 30px;
        background: #e0e0e0;
        border-radius: 8px;
        margin: 15px 0;
        position: relative;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
    }
    .confidence-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%,-50%);
        color: white;
        font-weight: bold;
    }
    .safe { color: #4CAF50; }
    .danger { color: #F44336; }
    .face-box {
        border: 3px solid #1E88E5;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Model loading with error handling
@st.cache_resource
def load_mask_model():
    try:
        os.makedirs("models", exist_ok=True)
        model_path = hf_hub_download(
            repo_id="HamzaNawaz17/Mask_Detection_system",
            filename="mask_detection_model.h5",
            cache_dir="models"
        )
        custom_objects = {"InputLayer": tf.keras.layers.InputLayer}
        model = load_model(model_path, custom_objects=custom_objects)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def detect_faces(image_np):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                ih, iw, _ = image_np.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                x, y, w, h = bbox
                face = image_np[y:y+h, x:x+w]
                faces.append((face, bbox))
        
        return faces, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def preprocess_face(face_img):
    face_img = Image.fromarray(face_img).resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(face_img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def main():
    st.set_page_config(
        page_title="Smart Mask Detector",
        page_icon="😷",
        layout="wide"
    )
    
    st.markdown('<p class="header">😷 Smart Mask Detection System</p>', unsafe_allow_html=True)
    
    model = load_mask_model()
    if model is None:
        return
    
    tab1, tab2 = st.tabs(["Detection", "About"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image_np = np.array(Image.open(uploaded_file))
            faces, annotated_image = detect_faces(image_np)
            
            if not faces:
                st.warning("No faces detected in the image")
            else:
                st.image(annotated_image, caption="Detected Faces", use_column_width=True)
                
                for i, (face, bbox) in enumerate(faces):
                    with st.expander(f"Face {i+1} Analysis", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown('<div class="face-box">', unsafe_allow_html=True)
                            st.image(face, use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            face_array = preprocess_face(face)
                            prediction = model.predict(face_array)
                            confidence = float(prediction[0][0])
                            
                            if confidence < 0.5:
                                result = "✅ Mask Detected"
                                result_class = "safe"
                                conf_percent = round((1 - confidence) * 100, 2)
                                bar_color = "#4CAF50"
                                advice = "This person is following safety protocols."
                            else:
                                result = "⚠️ No Mask Detected"
                                result_class = "danger"
                                conf_percent = round(confidence * 100, 2)
                                bar_color = "#F44336"
                                advice = "Please wear a mask for safety."
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>Status: <span class="{result_class}">{result}</span></h3>
                                <p>Confidence: {conf_percent}%</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width:{conf_percent}%;background:{bar_color};"></div>
                                    <div class="confidence-text">{conf_percent}%</div>
                                </div>
                                <p>{advice}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ## About This App
        
        This system uses:
        - MediaPipe for face detection
        - TensorFlow for mask classification
        - Hugging Face Hub for model storage
        
        ### How It Works
        1. Upload an image containing faces
        2. The system detects all faces automatically
        3. Each face is analyzed for mask presence
        4. Results show detection confidence
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure faces are clearly visible
        - Masks should cover nose and mouth completely
        """)

if __name__ == "__main__":
    main()
