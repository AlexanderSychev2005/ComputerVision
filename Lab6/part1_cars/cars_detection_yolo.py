import cv2
import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(
    "yolov8m.pt"
)  #  s - small, m - medium, l - large, x - extra large, n - nano


INDEXES = [
    0,
    1,
    2,
    3,
    5,
    7,
]  # COCO class IDs for person, bicycle, car, motorcycle, bus, truck
CONFIDENCE = 0.4

cap = cv2.VideoCapture("../videos/video_cars_2k_1920_1080_30fps.mp4")


while True:
    timer = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, verbose=False, device=device, conf=CONFIDENCE, classes=INDEXES)

    for result in results:
        if len(result.boxes.xyxy):
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = model.names[class_id]
                confidence = round(float(box.conf), 3)

                if class_id == 0:
                    colour = (0, 255, 0)
                else:
                    colour = (255, 0, 0)


                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    2,
                )

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(
        frame,
        "FPS : " + str(int(fps)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLOv8 Car Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
