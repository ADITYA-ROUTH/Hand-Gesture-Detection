import cv2
import mediapipe as mp
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import urllib.request
import os
import av
import numpy as np
import threading

# --- MonkeyPatch for Streamlit-WebRTC shutdown on Python 3.13 ---
# Streamlit Cloud uses Python 3.13 by default, which can cause an AttributeError 
# during thread shutdown in streamlit-webrtc. This safe wrapper prevents the crash.
try:
    import streamlit_webrtc.shutdown
    if hasattr(streamlit_webrtc.shutdown, 'SessionShutdownObserver'):
        original_stop = streamlit_webrtc.shutdown.SessionShutdownObserver.stop
        
        def safe_stop(self, timeout: float = 1.0) -> None:
            try:
                original_stop(self, timeout)
            except Exception:
                pass
                
        streamlit_webrtc.shutdown.SessionShutdownObserver.stop = safe_stop
except Exception:
    pass

# --- Initialization & Setup ---
st.set_page_config(page_title="Hand Gesture Detection", layout="wide")
st.title("✋ Hand Gesture Detection Web App")
st.markdown("This application uses your **web browser's camera** to recognize hand gestures in real-time.")

# Mediapipe Tasks API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    with st.spinner("Downloading Hand Landmarker Model..."):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            model_path
        )

# --- Helper Functions ---
def fingers_up(landmarks):
    """
    Counts fingers up based on landmarks.
    Order: [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    
    # Thumb: compare x coordinates
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers: compare y coordinates of tip and PIP joint
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    
    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers


# --- webrtc_streamer Video Frame Callback ---

# Initialize landmarker globally for the callback
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Flip image horizontally for selfie-view
    img = cv2.flip(img, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands (Synchronous IMAGE mode for WebRTC frames)
    detection_result = landmarker.detect(mp_image)

    if detection_result and detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = img.shape
            
            # Draw landmarks
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            # Determine gesture
            fingers = fingers_up(hand_landmarks)
            gesture = ""
            
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Fist"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumb"
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "Index Finger"
            elif fingers == [0, 0, 1, 0, 0]:
                gesture = "Middle Finger"
            elif fingers == [0, 0, 0, 1, 0]:
                gesture = "Ring Finger"
            elif fingers == [0, 0, 0, 0, 1]:
                gesture = "Pinky Finger"
            elif fingers[1:] == [1, 1, 0, 0]:
                gesture = "Peace"
            elif sum(fingers) == 5 or sum(fingers[1:]) == 4:
                gesture = "Open Hand"
            else:
                gesture = f"Fingers up: {sum(fingers)}"

            # Draw text on image
            cv2.rectangle(img, (10, 10), (450, 80), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f'Gesture: {gesture}', (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- WebRTC STUN Server Configuration for Cloud Deployment ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Streamlit UI ---
st.markdown("### 🎥 Camera Feed")
st.write("Click **Start** below to allow permissions and begin gesture detection.")

webrtc_streamer(
    key="hand-gesture-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION
)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, OpenCV, and MediaPipe.")
