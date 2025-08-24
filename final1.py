import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Classes of interest
bag_classes = ["backpack", "handbag", "suitcase"]

# Open video
file_path = "video1.avi"
cap = cv2.VideoCapture(file_path)

# Video writer (original FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_original.avi", fourcc, fps, (width, height))  # same FPS

# Track static bags {id: (center, start_time, alerted)}
object_tracker = {}
object_id = 0
alert_time = 5  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    for *xyxy, conf, cls in detections:
        class_name = model.names[int(cls)]
        if class_name not in bag_classes:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        found = False
        for oid, (prev_center, start_time, alerted) in list(object_tracker.items()):
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))
            if dist < 40:  # same object
                object_tracker[oid] = (center, start_time, alerted)
                elapsed = time.time() - start_time
                if elapsed > alert_time and not alerted:
                    alert_text = f"⚠️ ALERT: Bag {oid} abandoned {int(elapsed)}s"
                    print(alert_text)
                    cv2.putText(frame, alert_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    object_tracker[oid] = (center, start_time, True)  # mark alerted
                found = True
                break

        if not found:
            object_id += 1
            object_tracker[object_id] = (center, time.time(), False)

        # Draw bounding box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bag {object_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Write annotated frame to video
    out.write(frame)

    # Show live
    cv2.imshow("Bag Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
