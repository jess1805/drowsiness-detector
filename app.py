import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="AI Driver Drowsiness Detector", layout="wide")

st.title("AI Driver Drowsiness & Yawn Detector")
st.write("This app uses your webcam to monitor fatigue in real-time.")

# Load the model and cascades
@st.cache_resource
def load_assets():
    # Make sure this name matches exactly what you uploaded
    model = load_model('eye_yawn_model_final.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_lefteye_2splits.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_smile.xml')
    return model, face_cascade, eye_cascade, smile_cascade

model, face, eye, smile = load_assets()

labels = ['Closed', 'Open', 'Yawn', 'No Yawn']

img_file_buffer = st.camera_input("Take a photo to check status")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    faces = face.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No face detected. Please adjust your camera.")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        prediction = model.predict(roi_gray)
        res = np.argmax(prediction)
        
        st.subheader(f"Detection Result: {labels[res]}")
        
        if res == 0 or res == 2:
            st.error("⚠️ ALERT: DROWSINESS OR YAWN DETECTED!")
        else:
            st.success("✅ Status: Alert and Awake")
