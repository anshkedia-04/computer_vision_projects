# Streamlit_apps/emotion_app.py
import streamlit as st
import subprocess

st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

st.title("😊 Real-Time Emotion Detection")
st.markdown("""
This app uses a webcam to detect human emotions using a CNN model trained on facial expression datasets.
- It can recognize **Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise**.
- Built with **TensorFlow, OpenCV, and Streamlit**.
""")

st.warning("This will open your webcam in a new window.")

if st.button("▶ Start Emotion Detection"):
    subprocess.Popen(["python", "emotion_camera.py"])
    st.info("Press 'q' in the camera window to quit.")
