'''
import streamlit as st
from pathlib import Path
import os
from main import detect_vehicle
from PIL import Image

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

st.title("🚨 Emergency Vehicle Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    image_path = Path("uploads/uploaded.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(Image.open(image_path), caption="Uploaded Image", use_container_width=True)

    # Run detection
    result_path, class_ids = detect_vehicle(str(image_path))

    # Show result
    st.image(result_path, caption="Detection Result", use_container_width=True)

    # Emergency logic
    emergency_classes = [0, 1, 2]  # class indices for Ambulance, Fire Truck, Police Vehicle
    if any(cls in emergency_classes for cls in class_ids):
        st.success("✅ Emergency vehicle detected! Override signal to GREEN.")
    else:
        st.info("🚗 No emergency vehicle detected. Signal stays RED.")
'''


import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from main import detect_vehicles
from audio_detector import detect_siren_audio

# -------------------------- Signal State Persistence --------------------------
SIGNAL_HOLD_DURATION = 5  # seconds
signal_last_changed_time = 0
last_signal_state = "🔴 Red Light"

# -------------------------- Streamlit Config --------------------------
st.set_page_config(page_title="Emergency Vehicle Prioritization", layout="wide")
st.title("🚨 Emergency Vehicle Detection and Signal Control System")

# -------------------------- Detection Mode --------------------------
mode = st.sidebar.radio("Select Input Mode", ("Upload Video", "Webcam"))
fusion_logic = st.sidebar.radio("Fusion Logic", ("AND", "OR"))
st.sidebar.markdown("---")
st.sidebar.markdown("Model: YOLOv8 + YAMNet")
st.sidebar.markdown("Direction Inference: ✅")
st.sidebar.markdown("Green Signal Persistence: ✅")

# -------------------------- File Upload or Webcam --------------------------
video_path = None
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    video_path = 0  # Webcam index

# -------------------------- Helper: Infer Direction --------------------------
def infer_direction(bboxes, frame_width):
    approaching_threshold = frame_width * 0.4
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        if center_x < approaching_threshold:
            return "Approaching"
    return "Not Approaching"

# -------------------------- Helper: Persistent Signal Control --------------------------
def simulate_signal(emergency_detected, direction_detected):
    global last_signal_state, signal_last_changed_time

    current_time = time.time()

    if emergency_detected and direction_detected == "Approaching":
        last_signal_state = "🔓 Green Light (EMERGENCY PRIORITY)"
        signal_last_changed_time = current_time
    elif current_time - signal_last_changed_time < SIGNAL_HOLD_DURATION:
        pass  # maintain previous green state
    else:
        last_signal_state = "🔴 Red Light"

    return last_signal_state

# -------------------------- Start Detection --------------------------
if video_path is not None:
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30

    # Detect siren from audio (one-time)
    audio_siren_detected = detect_siren_audio(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Run visual detection
        vehicle_classes, emergency_visual_detected, annotated_frame, bboxes = detect_vehicles(frame_resized)

        # Direction
        direction = infer_direction(bboxes, frame_resized.shape[1])

        # Fusion Logic
        if fusion_logic == "AND":
            emergency_final = emergency_visual_detected and audio_siren_detected
        else:
            emergency_final = emergency_visual_detected or audio_siren_detected

        # Signal Control
        signal_status = simulate_signal(emergency_final, direction)

        # Display
        status_text = f"🚨 Emergency Detected: {'Yes' if emergency_final else 'No'}"
        direction_text = f"📍 Direction: {direction}"
        signal_text = f"🚦 Signal Status: {signal_status}"

        stframe.image(annotated_frame, caption=f"{status_text} | {direction_text} | {signal_text}", channels="BGR", use_container_width=True)

        # Optional: Add delay to simulate real-time
        if mode == "Webcam":
            cv2.waitKey(1)
        else:
            cv2.waitKey(wait_time)

    cap.release()
else:
    st.info("Please upload a video or enable webcam to start detection.")
