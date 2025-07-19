import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="🧠 ASL Sign Detection", layout="centered")
st.title("🧠 ASL Sign Detection")
st.markdown("This app uses a trained CNN model to detect American Sign Language letters from webcam input.")

# Load the trained model
model_path = "asl_sign_language_model.h5"
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure 'asl_sign_language_model.h5' is in this directory.")
    st.stop()

model = load_model(model_path)

# Hardcoded class labels (same order as training set)
class_labels = [chr(i) for i in range(65, 91)]  # A-Z

run = st.button("Start Camera")

if run:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)) / 255.0
        img_array = np.expand_dims(resized, axis=(0, -1))

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        gesture = class_labels[predicted_class]

        cv2.putText(frame, f'Prediction: {gesture}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
