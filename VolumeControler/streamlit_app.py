import streamlit as st
import cv2
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from HandTrackingModule import handDetector

# Streamlit Page Config
st.set_page_config(page_title="🎚 Volume Control using Hand", layout="centered")

# Title
st.title("🎚 Hand Gesture Volume Control")
st.markdown("Control your system volume using finger distance in front of your webcam.")

# Session state to handle start/stop
if "run_volume" not in st.session_state:
    st.session_state.run_volume = False

# Buttons
start_btn = st.button("Start", key="start_button")
stop_btn = st.button("Stop", key="stop_button")

if start_btn:
    st.session_state.run_volume = True
if stop_btn:
    st.session_state.run_volume = False

# Run Volume Control Logic
if st.session_state.run_volume:
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    detector = handDetector(detectionCon=0.7)

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]
    vol = 0
    volBar = 400
    volPer = 0

    stframe = st.empty()

    while st.session_state.run_volume:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
