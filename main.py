import cv2
import mediapipe as mp
import time

# Mediapipe Tasks API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables for async callback
latest_result = None

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Download the model if not exists
import urllib.request
import os
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=1
)

def fingers_up(landmarks):
    """
    Counts fingers up based on landmarks.
    Order: [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    
    # Thumb: compare x coordinates (assuming right hand for simplicity, logic can be complex for both)
    # Simple heuristic: if thumb tip is further right than thumb IP joint
    # Need to check handedness for robust thumb detection, but this is a simplified version
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

def main():
    # To use your phone's camera, download the "IP Webcam" app (Android) or "EpocCam" (iOS)
    # 1. Connect your phone and PC to the SAME WiFi network.
    # 2. Start the server on the app. It will give you an IPv4 address (e.g., http://192.168.1.5:8080)
    # 3. Replace the `0` below with your full IP Webcam URL + "/video"
    # Example: cap = cv2.VideoCapture("http://192.168.1.5:8080/video")
    
    cap = cv2.VideoCapture("http://192.168.0.175:8080/video") # Change this!
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    global latest_result

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            # and convert the BGR image to RGB.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send live image data to perform hand landmarking
            # The async callback will update latest_result
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            if latest_result is not None and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw landmarks simply (we can't use mp.solutions.drawing_utils)
                    h, w, _ = frame.shape
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    
                    # Get fingers up
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

                    cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, f'Gesture: {gesture}', (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            cv2.imshow('Hand Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
