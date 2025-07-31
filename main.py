'''
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("EmergencyVehicle/detector_model/weights/best.pt")

def detect_vehicle(image_path):
    results = model.predict(source=image_path, save=True, conf=0.3)
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    result_path = list(Path(results[0].save_dir).glob("*.jpg"))[0]
    return result_path, class_ids
'''

import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("EmergencyVehicle/detector_model/weights/best.pt")  # Use your fine-tuned model if available

# Define emergency vehicle classes (ensure your model has these labels)
EMERGENCY_CLASSES = {"ambulance", "fire_truck", "police"}

def detect_vehicles(frame):
    """
    Perform vehicle detection on a single frame.

    Args:
        frame (numpy.ndarray): The input image frame.

    Returns:
        tuple: (
            vehicle_classes: List of class names detected,
            emergency_detected: Boolean indicating if emergency vehicle is present,
            annotated_frame: Image with bounding boxes drawn,
            bboxes: List of bounding boxes [x1, y1, x2, y2]
        )
    """
    results = model.predict(frame, verbose=False)

    vehicle_classes = []
    emergency_detected = False
    bboxes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()

            # Append class name
            vehicle_classes.append(cls_name)

            # Extract bounding box in [x1, y1, x2, y2] format
            xyxy = box.xyxy[0].tolist()
            bboxes.append(xyxy)

            # Emergency detection
            if any(em_cls in cls_name for em_cls in EMERGENCY_CLASSES):
                emergency_detected = True

    # Annotated frame (with YOLO boxes and labels)
    annotated_frame = results[0].plot()

    return vehicle_classes, emergency_detected, annotated_frame, bboxes
