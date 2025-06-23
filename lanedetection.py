import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model (ensure it's trained on lane data or use a custom model)
model = YOLO("yolov8n.pt")  # Replace with your lane detection model if available

# ip webcam url
url = "http://10.2.25.103:8080/video"

# Open video or webcam
cap = cv2.VideoCapture(url)  # Use 0 for webcam or 'your_video.mp4'

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1200, 800))  # Adjust resolution as needed

    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]

    # Draw YOLO detections (bounding boxes, masks, etc.)
    annotated_frame = results.plot()

    # Display result
    cv2.imshow("Lane Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
