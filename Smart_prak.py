import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Streamlit UI
st.title("Smart Parking System 🚗")
st.write("Real-time vehicle detection using YOLOv8.")

# Placeholder for the video feed
frame_placeholder = st.empty()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Camera could not be opened. Check permissions!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to grab frame.")
        break

    # Run YOLO detection
    results = model(frame)

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert to RGB (Streamlit needs RGB format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display in Streamlit
    frame_placeholder.image(frame, channels="RGB", use_column_width=True)

cap.release()
cv2.destroyAllWindows()
