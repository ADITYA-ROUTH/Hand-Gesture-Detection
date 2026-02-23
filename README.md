# ✋ Hand Gesture Detection Web App

A lightweight, real-time hand gesture detection application built with Python, OpenCV, and MediaPipe. 

This project tracks a single hand using your webcam, counts raised fingers, and instantly classifies simple gestures like **Fist**, **Point**, **Peace**, and **Open Hand**. It has been fully converted into a web application using **Streamlit** and `streamlit-webrtc`, ready to be deployed to the cloud!

## Features
- **Real-time Tracking**: Uses the powerful [MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) for fast, robust hand landmarking.
- **Finger Counting Logic**: Accurately detects individual fingers (Thumb, Index, Middle, Ring, Pinky) using simple geometric heuristics.
- **Web-Ready**: Hosted completely in the browser without requiring a local window, utilizing `streamlit-webrtc` to pass video frames.

---

## 🛠️ Tech Stack
- **Python 3.13** (Core programming language)
- **OpenCV** (`opencv-python-headless`): Computer vision library for image manipulation and frame processing.
- **MediaPipe Tasks Vision API**: Machine learning pipeline for robust real-time hand landmark detection.
- **Streamlit**: Web application framework for generating rapid front-end UI.
- **Streamlit-WebRTC**: Enables real-time video streaming capabilities over WebRTC for the browser.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- A working webcam or IP camera application on your phone.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit development server:
```bash
streamlit run app.py
```
Your browser will automatically open to `http://localhost:8501`. Grant the site permission to access your webcam and click "Start" to begin gesture detection!

---

## ☁️ How to Deploy on Streamlit Community Cloud

This project is perfectly configured to run on Streamlit Community Cloud.

1. Ensure this entire project is uploaded to a public repository on your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select your GitHub repository and branch (`main`).
5. Set the **Main file path** to `app.py`.
6. Click **Deploy!**

*(Note: The `packages.txt` file handles the Linux-level `libgl1` and `libglib2.0-0` graphical dependencies required by OpenCV in cloud environments automatically).*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
