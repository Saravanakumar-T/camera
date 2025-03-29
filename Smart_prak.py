import cv2
import torch
import csv
import os
import datetime
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define parking lots with locations
PARKING_LOTS = {
    "Lot_1": {"latitude": 13.0827, "longitude": 80.2707},
    "Lot_2": {"latitude": 13.0455, "longitude": 80.2358},
    "Lot_3": {"latitude": 13.0358, "longitude": 80.2109},
}

# 🚀 **Ask User to Select a Parking Lot**
print("\n🚗 Select the Parking Lot for this Camera:")
for lot in PARKING_LOTS.keys():
    print(f"🔹 {lot}")

selected_lot = input("\nEnter Parking Lot (Lot_1, Lot_2, Lot_3): ").strip()

# Ensure valid selection
if selected_lot not in PARKING_LOTS:
    print("❌ Invalid parking lot! Defaulting to Lot_1.")
    selected_lot = "Lot_1"

# Get selected parking lot details
selected_lat = PARKING_LOTS[selected_lot]["latitude"]
selected_lon = PARKING_LOTS[selected_lot]["longitude"]

# Define CSV file for the selected lot
CSV_FILE = f"{selected_lot}.csv"

print(f"\n✅ Camera assigned to {selected_lot} at ({selected_lat}, {selected_lon})")
print(f"📁 Data will be saved in: {CSV_FILE}")

# Function to initialize CSV for selected parking lot
def initialize_csv(csv_file):
    """Creates a CSV file for the selected lot if it does not exist."""
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Vehicle Type", "Vehicle Model", "Action", "Date", "Time", "Parking Lot ID", "Latitude", "Longitude"])

def log_vehicle(vehicle_type, vehicle_model, action):
    """Logs vehicle entry/exit into the respective lot's CSV file."""
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([vehicle_type, vehicle_model, action, date, time, selected_lot, selected_lat, selected_lon])

    print(f"Logged: {vehicle_type} {vehicle_model} {action} at {time} on {date}, {selected_lot}")

# Initialize CSV for the selected lot
initialize_csv(CSV_FILE)

# Define entry and exit zones
ENTRY_ZONE = (100, 200, 300, 400)
EXIT_ZONE = (400, 500, 600, 700)

# Track vehicle positions
vehicle_positions = defaultdict(lambda: None)
vehicle_count = 0  # Track number of vehicles inside

# Open camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Detect vehicles

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                prev_position = vehicle_positions[label]
                vehicle_positions[label] = center_y

                if prev_position is not None:
                    if prev_position < ENTRY_ZONE[1] and center_y >= ENTRY_ZONE[1]:
                        log_vehicle(label, "Unknown Model", "Entry")
                        vehicle_count += 1
                    elif prev_position > EXIT_ZONE[1] and center_y <= EXIT_ZONE[1]:
                        log_vehicle(label, "Unknown Model", "Exit")
                        vehicle_count -= 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicles Inside: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Smart Parking System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
