import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file
cap = cv2.VideoCapture("../Videos/f5.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define HSV temperature ranges (Celsius)
temperature_map = {
    "1200°C": [(0, 0, 250), (180, 50, 255)],   # White-Hot (Ultra Hot)
    "1000°C": [(45, 200, 250), (60, 255, 255)],  # Intense Yellow-White
    "900°C": [(0, 200, 200), (5, 255, 255)],    # Deep Red
    "700°C": [(5, 200, 200), (15, 255, 255)],  # Bright Red
    "600°C": [(15, 200, 200), (25, 255, 255)],  # Orange-Red (NO BOX)
    "500°C": [(25, 200, 200), (35, 255, 255)],  # Bright Orange (NO BOX)
}

def detect_temperature(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result_frame = frame.copy()

    for temp_label, (lower, upper) in temperature_map.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # **Filter only bright & hot regions**
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # **Find only large and small molten regions**
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # 🔹 Detect small areas **ONLY for 900°C, 1000°C, 1200°C**
            if temp_label in ["900°C", "1000°C", "1200°C"]:
                min_area = 50   # Detect small molten streams
            else:
                min_area = 1200  # Ignore small regions for lower temperatures

            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)

                # **Ensure white-hot is surrounded by molten metal colors**
                if temp_label == "1200°C":
                    surrounding_area = hsv_frame[max(y - 10, 0):min(y + h + 10, hsv_frame.shape[0]),
                                                 max(x - 10, 0):min(x + w + 10, hsv_frame.shape[1])]

                    contains_molten_colors = any(
                        cv2.inRange(surrounding_area, np.array(temperature_map[t][0], dtype=np.uint8),
                                    np.array(temperature_map[t][1], dtype=np.uint8)).any()
                        for t in ["1000°C", "900°C", "700°C"]
                    )

                    if not contains_molten_colors:
                        continue  # Skip if it's just a bright white spot with no molten surroundings

                # **Skip drawing boxes for < 700°C**
                if temp_label in ["600°C", "500°C"]:
                    continue

                # Draw bounding box in GREEN
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display temperature label in GREEN
                cv2.putText(result_frame, f"{temp_label}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result_frame

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Resize video for better processing
    img = cv2.resize(img, (1280, 720))

    # Apply temperature detection
    processed_frame = detect_temperature(img)

    cv2.imshow("Molten Metal Temperature Detection", processed_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
