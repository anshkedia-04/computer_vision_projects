# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp

# st.set_page_config(page_title="✍️ Air Writing", layout="centered")
# st.title("✍️ Air Writing with Hand Gestures")
# st.markdown("Draw in the air using your index finger. Use thumb + index pinch gesture to clear.")

# run = st.button("Start Drawing")

# if run:
#     # MediaPipe setup
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
#     mp_draw = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(0)
#     canvas = np.zeros((480, 640, 3), dtype=np.uint8)
#     prev_x, prev_y = 0, 0
#     draw_color = (255, 255, 255)
#     pinch_counter = 0
#     PINCH_THRESHOLD = 15
#     space_counter = 0
#     SPACE_THRESHOLD = 10
#     stframe = st.empty()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_frame)

#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 index_finger_tip = hand_landmarks.landmark[8]
#                 thumb_tip = hand_landmarks.landmark[4]
#                 h, w, _ = frame.shape
#                 x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
#                 thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

#                 if abs(x - thumb_x) < 30 and abs(y - thumb_y) < 30:
#                     pinch_counter += 1
#                     if pinch_counter > PINCH_THRESHOLD:
#                         canvas = np.zeros((480, 640, 3), dtype=np.uint8)
#                         prev_x, prev_y = 0, 0
#                 else:
#                     pinch_counter = 0
#                     if abs(prev_x - x) < 5 and abs(prev_y - y) < 5:
#                         space_counter += 1
#                         if space_counter > SPACE_THRESHOLD:
#                             prev_x, prev_y = 0, 0
#                     else:
#                         space_counter = 0
#                         if prev_x != 0 and prev_y != 0:
#                             cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
#                         prev_x, prev_y = x, y

#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         else:
#             prev_x, prev_y = 0, 0

#         blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
#         stframe.image(blended, channels="RGB")

#     cap.release()
#     cv2.destroyAllWindows()


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

# Page configuration with dark theme
st.set_page_config(
    page_title="Air Writing Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e2936 0%, #2d3748 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Stats container */
    .stats-container {
        background: #1e2936;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #2d3748 0%, #1e2936 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #3b82f6;
    }
    
    .stat-value {
        color: #3b82f6;
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Instructions box */
    .instructions {
        background: #1e2936;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 3px solid #10b981;
        margin: 1.5rem 0;
    }
    
    .instruction-title {
        color: #10b981;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .instruction-item {
        color: #cbd5e1;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .instruction-item:before {
        content: "→";
        position: absolute;
        left: 0;
        color: #3b82f6;
    }
    
    /* Status indicator */
    .status-active {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border-radius: 20px;
        border: 1px solid #10b981;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-inactive {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border-radius: 20px;
        border: 1px solid #ef4444;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #1e2936;
    }
    
    /* Remove default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Alert box */
    .alert-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        color: #93c5fd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">✍️ Air Writing Studio</h1>
    <p class="subtitle">Professional gesture-based writing interface powered by computer vision</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("---")
    st.markdown("**Drawing Settings**")
    line_thickness = st.slider("Line Thickness", 1, 15, 5)
    opacity = st.slider("Canvas Opacity", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("**Detection Settings**")
    detection_conf = st.slider("Detection Confidence", 0.5, 1.0, 0.8, 0.05)
    tracking_conf = st.slider("Tracking Confidence", 0.5, 1.0, 0.8, 0.05)
    
    st.markdown("---")
    st.markdown("**Advanced**")
    pinch_sensitivity = st.slider("Clear Gesture Sensitivity", 10, 50, 30)
    space_threshold = st.slider("Space Detection", 5, 20, 10)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #3b82f6;'>
        <p style='color: #93c5fd; margin: 0; font-size: 0.9rem;'>
            <strong>💡 Tip:</strong> Adjust settings to optimize for your environment and lighting conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Instructions section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="instructions">
        <div class="instruction-title">📋 How to Use</div>
        <div class="instruction-item">Position your hand in front of the camera</div>
        <div class="instruction-item">Extend your index finger to draw in the air</div>
        <div class="instruction-item">Pinch thumb and index finger together to clear canvas</div>
        <div class="instruction-item">Hold finger still to create spaces between strokes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-container">
        <div class="stat-box">
            <span class="stat-value">🎯</span>
            <div class="stat-label">Ready to Draw</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Control buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    start_button = st.button("🚀 Start Drawing Session", use_container_width=True)

if start_button:
    # Status indicator
    st.markdown('<div class="status-active">● ACTIVE SESSION</div>', unsafe_allow_html=True)
    
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("❌ Failed to access camera. Please check your camera permissions.")
        st.stop()
    
    # Get camera dimensions
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        h, w = 480, 640
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    prev_x, prev_y = 0, 0
    draw_color = (0, 200, 255)  # Cyan color for drawing
    pinch_counter = 0
    space_counter = 0
    
    # Create placeholder for video feed
    stframe = st.empty()
    
    # Stats placeholders
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    fps_placeholder = stats_col1.empty()
    hand_status_placeholder = stats_col2.empty()
    mode_placeholder = stats_col3.empty()
    
    # Stop button
    stop_button = st.button("⏹️ Stop Session", type="secondary")
    
    frame_count = 0
    start_time = time.time()
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera feed interrupted")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        hand_detected = False
        current_mode = "Idle"
        
        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # Check for pinch gesture (clear canvas)
                distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)
                if distance < pinch_sensitivity:
                    pinch_counter += 1
                    current_mode = "Clear Gesture"
                    if pinch_counter > 15:
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        prev_x, prev_y = 0, 0
                        # Visual feedback for clearing
                        cv2.putText(frame, "CANVAS CLEARED", (w//2 - 100, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    pinch_counter = 0
                    
                    # Check for stationary finger (space)
                    if prev_x != 0 and abs(prev_x - x) < 5 and abs(prev_y - y) < 5:
                        space_counter += 1
                        current_mode = "Space Detection"
                        if space_counter > space_threshold:
                            prev_x, prev_y = 0, 0
                    else:
                        space_counter = 0
                        current_mode = "Drawing"
                        # Draw line
                        if prev_x != 0 and prev_y != 0:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, line_thickness)
                        prev_x, prev_y = x, y
                
                # Draw hand landmarks with custom colors
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)
                )
                
                # Draw cursor indicator
                cv2.circle(frame, (x, y), 10, (0, 200, 255), -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
        else:
            prev_x, prev_y = 0, 0
        
        # Blend frame with canvas
        blended = cv2.addWeighted(frame, 1 - opacity, canvas, opacity, 0)
        
        # Add professional overlay
        overlay = blended.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (30, 41, 54), -1)
        cv2.addWeighted(overlay, 0.7, blended, 0.3, 0, blended)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
        else:
            fps = 0
        
        # Add text overlays
        if fps > 0:
            cv2.putText(blended, f"FPS: {int(fps)}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(blended, f"Mode: {current_mode}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        stframe.image(blended, channels="RGB")
        
        # Update stats
        fps_placeholder.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{int(fps) if fps > 0 else '--'}</span>
            <div class="stat-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)
        
        hand_status_placeholder.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{'✓' if hand_detected else '✗'}</span>
            <div class="stat-label">Hand Detected</div>
        </div>
        """, unsafe_allow_html=True)
        
        mode_placeholder.markdown(f"""
        <div class="stat-box">
            <span class="stat-value" style="font-size: 1rem;">{current_mode}</span>
            <div class="stat-label">Current Mode</div>
        </div>
        """, unsafe_allow_html=True)
    
    cap.release()
    cv2.destroyAllWindows()
    st.markdown('<div class="status-inactive">● SESSION ENDED</div>', unsafe_allow_html=True)
    st.success("✅ Drawing session completed successfully!")
else:
    st.markdown('<div class="status-inactive">● INACTIVE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-box">
        <strong>ℹ️ Ready to begin:</strong> Click the "Start Drawing Session" button above to launch the camera and begin drawing in the air.
    </div>
    """, unsafe_allow_html=True)