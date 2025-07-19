import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="✍️ Air Writing", layout="centered")
st.title("✍️ Air Writing with Hand Gestures")
st.markdown("Draw in the air using your index finger. Use thumb + index pinch gesture to clear.")

run = st.button("Start Drawing")

if run:
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_x, prev_y = 0, 0
    draw_color = (255, 255, 255)
    pinch_counter = 0
    PINCH_THRESHOLD = 15
    space_counter = 0
    SPACE_THRESHOLD = 10
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                if abs(x - thumb_x) < 30 and abs(y - thumb_y) < 30:
                    pinch_counter += 1
                    if pinch_counter > PINCH_THRESHOLD:
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        prev_x, prev_y = 0, 0
                else:
                    pinch_counter = 0
                    if abs(prev_x - x) < 5 and abs(prev_y - y) < 5:
                        space_counter += 1
                        if space_counter > SPACE_THRESHOLD:
                            prev_x, prev_y = 0, 0
                    else:
                        space_counter = 0
                        if prev_x != 0 and prev_y != 0:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
                        prev_x, prev_y = x, y

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = 0, 0

        blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        stframe.image(blended, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
