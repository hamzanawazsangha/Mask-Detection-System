import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gdown
import os

# Set page config
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Drive model download function
@st.cache_resource
def load_model():
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1yREfq0xN6pglc9Bo3yMj8qm8RZM4dY-Y"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Local path to save the model
    model_path = "mask_detection_model.h5"
    
    # Download if not exists
    if not os.path.exists(model_path):
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        model.make_predict_function()  # Optimize for inference
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# Function to save comments
def save_comment(name, email, comment):
    with open("comments.txt", "a") as f:
        f.write(f"Name: {name}\nEmail: {email}\nComment: {comment}\n{'='*30}\n")

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 20px !important;
        color: #0E1117 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #FAFAFA;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #E3F2FD;
    }
    .safe {
        color: #4CAF50;
        font-weight: bold;
    }
    .danger {
        color: #F44336;
        font-weight: bold;
    }
    .confidence-bar {
        height: 25px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #E0E0E0;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
    }
    .confidence-text {
        position: absolute;
        top: 50%;
        left: 10px;
        transform: translateY(-50%);
        color: white;
        font-weight: bold;
    }
    .info-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
        background-color: #FFF8E1;
    }
    .feedback-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
        background-color: #E8F5E9;
    }
</style>
""", unsafe_allow_html=True)

# Preprocess image function
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Main app
def main():
    st.markdown('<p class="header">😷 Mask Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image to check if the person is wearing a mask</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and model is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess the image
            img_array = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            
            # Determine result
            if confidence < 0.5:
                result = "Mask Detected 😷"
                result_class = "safe"
                conf_percent = round((1 - confidence) * 100, 2)
            else:
                result = "No Mask Detected ❌"
                result_class = "danger"
                conf_percent = round(confidence * 100, 2)
            
            # Display results
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            st.markdown(f"**Status:** <span class='{result_class}'>{result}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {conf_percent}%")
            
            # Confidence bar
            bar_color = "#4CAF50" if result_class == "safe" else "#F44336"
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {conf_percent}%; background-color: {bar_color};"></div>
                <div class="confidence-text">{conf_percent}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("About Mask Detection")
        st.write("""
        This system uses a deep learning model to detect whether a person in an image is:
        
        - **Wearing a mask** (safe, recommended)
        - **Not wearing a mask** (potential risk)
        
        The model was trained on facial images and outputs a confidence score between 0 (mask) and 1 (no mask).
        """)
        
        st.subheader("How to Use")
        st.write("""
        1. Upload a clear facial image (JPG, JPEG, or PNG format)
        2. The system will analyze the image
        3. View the prediction results with confidence percentage
        """)
        
        st.subheader("Tips for Best Results")
        st.write("""
        - Use clear, well-lit images
        - Ensure the face is clearly visible
        - Avoid images with multiple faces
        - Center the face in the image
        """)
        
        st.subheader("Disclaimer")
        st.write("""
        This tool is for informational purposes only. Always follow local health guidelines 
        and regulations regarding mask usage.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feedback/Suggestion Section
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.subheader("We Value Your Feedback!")
        
        with st.form("feedback_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email (optional)")
            comment = st.text_area("Your Feedback or Suggestions")
            
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                if comment:  # At least require a comment
                    save_comment(name, email, comment)
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please provide your feedback before submitting")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
