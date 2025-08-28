import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from main import load_model, predict_image_class

st.set_page_config(page_title="Waste Classification CNN Convolutional Neural Network", page_icon="♻️")

# Cache the model loading function
@st.cache_resource
def get_model():
    return load_model()

st.title("♻️ Waste Classification CNN Convolutional Neural Network")

st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select classification mode:", ["Upload/Capture Image", "Real-Time Webcam"])

st.header("Introduction")
st.write("""
This project utilizes a Convolutional Neural Network (CNN) to classify waste into four categories: **Biodegradable, Non-Biodegradable, Trash, or Hazardous**. The model is designed to assist in waste management by automating the classification process, making it easier to sort and recycle waste effectively.

### Resource:
- **Data-Set**: "kaggle datasets download -d alistairking/recyclable-and-household-waste-classification".

### Key Features:
- **Model Loading**: The model is loaded using a cached function to improve performance.
- **Image Upload**: Users can upload images of waste through the Streamlit interface.
- **Prediction**: The uploaded image is processed and classified into one of the four categories using the trained CNN model.
- **User Feedback**: The app provides visual feedback by displaying the uploaded image and the predicted class.
""")

model = get_model()
if model is None:
    st.error("Model not loaded. Please check your model file.")
    st.stop()

if mode == "Upload/Capture Image":
    st.write("Upload an image of waste, and the model will classify it into one of four categories: **Biodegradable, Non-Biodegradable, Trash, or Hazardous.**")
    uploaded_image = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])
    st.write("Or capture an image using your webcam:")
    camera_image = st.camera_input("📷 Take a photo")

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            predicted_class = predict_image_class(image, model)
            st.balloons()
            st.success(f"✅ **Predicted Class:** *{predicted_class}*")
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert('RGB')
            st.image(image, caption="Captured Image", use_container_width=True)
            predicted_class = predict_image_class(image, model)
            st.balloons()
            st.success(f"✅ **Predicted Class:** *{predicted_class}*")
        except Exception as e:
            st.error(f"❌ Error processing the camera image: {e}")

elif mode == "Real-Time Webcam":
    st.write("""
    ## Real-Time Webcam Classification
    This mode uses your webcam to classify waste in real-time. Press 'Stop' to end the session.
    """)
    run_webcam = st.button("Start Webcam Classification")
    stop_webcam = st.button("Stop Webcam")
    stframe = st.empty()
    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                predicted_class = predict_image_class(pil_image, model)
                cv2.putText(frame, f"Class: {predicted_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.05)
                if stop_webcam:
                    break
            cap.release()
