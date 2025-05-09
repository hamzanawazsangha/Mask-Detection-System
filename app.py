import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import mediapipe as mp
from huggingface_hub import hf_hub_download
import os

# Set page config with better visuals
st.set_page_config(
    page_title="Smart Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## A sophisticated mask detection system using face detection and deep learning"
    }
)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Custom CSS for enhanced styling
st.markdown("""
<style>
    :root {
        --primary: #1E88E5;
        --safe: #4CAF50;
        --danger: #F44336;
        --warning: #FFC107;
    }
    
    .header {
        font-size: 42px !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #1E88E5, #0D47A1);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        text-align: center;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 2px solid #E0E0E0;
    }
    
    .subheader {
        font-size: 22px !important;
        color: #424242 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .upload-box {
        border: 2px dashed var(--primary);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #FAFAFA;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.2);
        transform: translateY(-2px);
    }
    
    .result-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .safe {
        color: var(--safe);
        font-weight: 800;
    }
    
    .danger {
        color: var(--danger);
        font-weight: 800;
    }
    
    .confidence-bar-container {
        height: 30px;
        border-radius: 8px;
        margin: 20px 0;
        background-color: #E0E0E0;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    .confidence-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    .info-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        background: linear-gradient(145deg, #fff9e6, #fff3e0);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    .feedback-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        background: linear-gradient(145deg, #e8f5e9, #e0f7fa);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    .face-box {
        border: 3px solid var(--primary);
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #1E88E5, #0D47A1);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }
    
    .tab-content {
        padding: 20px 0;
    }
    
    @media (max-width: 768px) {
        .header {
            font-size: 32px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download model from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="HamzaNawaz17/Mask_Detection_system",
            filename="mask_detection_model.h5",
            cache_dir="models"
        )
        
        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)
        model.make_predict_function()  # Optimize for inference
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Function to detect faces using MediaPipe
@st.cache_resource
def init_face_detection():
    return mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

def detect_faces(image_np, face_detection):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            # Get face bounding box
            ih, iw, _ = image_np.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih)
            )
            
            # Extract face ROI
            x, y, w, h = bbox
            face = image_np[y:y+h, x:x+w]
            faces.append((face, bbox))
    
    return faces, image_rgb

# Function to preprocess face image for mask prediction
def preprocess_face(face_img):
    face_img = Image.fromarray(face_img)
    face_img = face_img.resize((256, 256))
    img_array = image.img_to_array(face_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Function to save feedback
def save_feedback(name, email, comment):
    with open("feedback.txt", "a") as f:
        f.write(f"Name: {name}\nEmail: {email}\nComment: {comment}\n{'='*40}\n")

# Main application
def main():
    st.markdown('<p class="header">😷 Smart Mask Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image to detect faces and check for mask compliance</p>', unsafe_allow_html=True)
    
    # Initialize models
    face_detection = init_face_detection()
    model = load_model()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Mask Detection", "How It Works", "Feedback"])
    
    with tab1:
        st.markdown("""
        <div class='tab-content'>
            <h3 style='color: #1E88E5; margin-bottom: 20px;'>Upload an Image for Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file is not None and model is not None:
                # Read the image
                image_np = np.array(Image.open(uploaded_file))
                
                # Detect faces
                faces, annotated_image = detect_faces(image_np, face_detection)
                
                if not faces:
                    st.warning("No faces detected in the image. Please try another image with clear faces.")
                else:
                    # Display original image with face detections
                    st.image(
                        annotated_image, 
                        caption='Detected Faces', 
                        use_column_width=True,
                        channels="RGB"
                    )
                    
                    # Process each detected face
                    for i, (face, bbox) in enumerate(faces):
                        st.markdown(f"### Face {i+1} Analysis")
                        
                        # Display the cropped face
                        col_face1, col_face2 = st.columns([1, 2])
                        
                        with col_face1:
                            st.markdown('<div class="face-box">', unsafe_allow_html=True)
                            st.image(face, caption=f'Face {i+1}', use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_face2:
                            # Preprocess and predict
                            face_array = preprocess_face(face)
                            prediction = model.predict(face_array)
                            confidence = float(prediction[0][0])
                            
                            # Determine result
                            if confidence < 0.5:
                                result = "✅ Mask Detected"
                                result_class = "safe"
                                conf_percent = round((1 - confidence) * 100, 2)
                                advice = "Good job! You're following safety protocols."
                            else:
                                result = "⚠️ No Mask Detected"
                                result_class = "danger"
                                conf_percent = round(confidence * 100, 2)
                                advice = "Please wear a mask to protect yourself and others."
                            
                            # Display results
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="margin-bottom: 15px;">
                                <span style="font-size: 18px; font-weight: 600;">Status:</span> 
                                <span class="{result_class}" style="font-size: 20px;">{result}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="margin: 20px 0;">
                                <span style="font-size: 16px; font-weight: 600;">Confidence:</span> 
                                <span style="font-size: 18px;">{conf_percent}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence bar with animation
                            bar_color = "var(--safe)" if result_class == "safe" else "var(--danger)"
                            st.markdown(f"""
                            <div class="confidence-bar-container">
                                <div class="confidence-fill" style="width: {conf_percent}%; background-color: {bar_color};"></div>
                                <div class="confidence-text">{conf_percent}% Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="margin-top: 20px; padding: 12px; background-color: #f5f5f5; border-radius: 8px;">
                                <span style="font-size: 16px;">{advice}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #1E88E5; margin-bottom: 15px;'>Real-time Analysis</h3>
            <p style='font-size: 16px; line-height: 1.6;'>
            Our system first detects all faces in your image using advanced computer vision, 
            then analyzes each face individually for mask presence with deep learning.
            </p>
            
            <div style='margin: 25px 0;'>
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='width: 40px; height: 40px; background-color: #1E88E5; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 15px;'>
                        <span style='color: white; font-weight: bold;'>1</span>
                    </div>
                    <span style='font-size: 16px;'>Upload a clear image containing faces</span>
                </div>
                
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='width: 40px; height: 40px; background-color: #1E88E5; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 15px;'>
                        <span style='color: white; font-weight: bold;'>2</span>
                    </div>
                    <span style='font-size: 16px;'>System detects all faces automatically</span>
                </div>
                
                <div style='display: flex; align-items: center;'>
                    <div style='width: 40px; height: 40px; background-color: #1E88E5; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 15px;'>
                        <span style='color: white; font-weight: bold;'>3</span>
                    </div>
                    <span style='font-size: 16px;'>Get detailed analysis for each face</span>
                </div>
            </div>
            
            <h3 style='color: #1E88E5; margin: 25px 0 15px 0;'>Best Practices</h3>
            <ul style='font-size: 16px; line-height: 1.6; padding-left: 20px;'>
                <li>Use well-lit images for better accuracy</li>
                <li>Ensure faces are clearly visible</li>
                <li>For groups, position faces towards the camera</li>
                <li>Masks should cover nose and mouth completely</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class='tab-content'>
            <h2 style='color: #1E88E5;'>How Our Mask Detection System Works</h2>
            
            <div style='background-color: #f8f9fa; border-radius: 12px; padding: 25px; margin: 20px 0;'>
                <h3 style='color: #1E88E5;'>1. Face Detection</h3>
                <p style='font-size: 16px; line-height: 1.6;'>
                Using MediaPipe's advanced face detection technology, our system first identifies all human faces 
                in your uploaded image. This ensures we only analyze relevant regions for mask detection.
                </p>
                <div style='display: flex; justify-content: center; margin: 15px 0;'>
                    <img src='https://mediapipe.dev/images/mobile/face_detection_android_gpu.gif' style='width: 60%; border-radius: 8px;'>
                </div>
            </div>
            
            <div style='background-color: #f8f9fa; border-radius: 12px; padding: 25px; margin: 20px 0;'>
                <h3 style='color: #1E88E5;'>2. Mask Classification</h3>
                <p style='font-size: 16px; line-height: 1.6;'>
                Each detected face is then analyzed by our deep learning model trained on thousands of images 
                to accurately distinguish between masked and unmasked faces with high confidence.
                </p>
                <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0;'>
                    <div style='text-align: center; margin: 10px;'>
                        <div style='width: 120px; height: 120px; background-color: #E3F2FD; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;'>
                            <span style='font-size: 50px;'>😷</span>
                        </div>
                        <p style='font-weight: 600;'>Mask Detected</p>
                    </div>
                    <div style='text-align: center; margin: 10px;'>
                        <div style='width: 120px; height: 120px; background-color: #FFEBEE; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;'>
                            <span style='font-size: 50px;'>🙅‍♂️</span>
                        </div>
                        <p style='font-weight: 600;'>No Mask</p>
                    </div>
                </div>
            </div>
            
            <div style='background-color: #f8f9fa; border-radius: 12px; padding: 25px; margin: 20px 0;'>
                <h3 style='color: #1E88E5;'>3. Detailed Results</h3>
                <p style='font-size: 16px; line-height: 1.6;'>
                For each face, you receive a clear classification with confidence percentage and visual feedback 
                to help you understand the results.
                </p>
                <div style='margin: 20px 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                        <div style='width: 30px; height: 30px; background-color: #4CAF50; border-radius: 50%; 
                                    margin-right: 15px;'></div>
                        <span style='font-size: 16px;'>Green indicates mask detected with high confidence</span>
                    </div>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 30px; height: 30px; background-color: #F44336; border-radius: 50%; 
                                    margin-right: 15px;'></div>
                        <span style='font-size: 16px;'>Red indicates no mask detected</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class='tab-content'>
            <h2 style='color: #1E88E5;'>We Value Your Feedback</h2>
            <p style='font-size: 16px; line-height: 1.6; margin-bottom: 25px;'>
            Your input helps us improve this system. Please share your experience, suggestions, 
            or report any issues you encountered.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("feedback_form"):
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                name = st.text_input("Your Name", placeholder="John Doe")
            with col_f2:
                email = st.text_input("Your Email (optional)", placeholder="john@example.com")
            
            comment = st.text_area(
                "Your Feedback", 
                placeholder="What did you like? What can we improve?",
                height=150
            )
            
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                if comment.strip():
                    save_feedback(name, email, comment)
                    st.success("Thank you for your valuable feedback! We appreciate your time.")
                else:
                    st.warning("Please provide your feedback before submitting")

if __name__ == "__main__":
    main()
