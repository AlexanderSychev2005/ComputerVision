import cv2
import torch
from ultralytics import YOLO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture("../videos/people_cows_1920_1080_30fps.mp4")
while True:
    timer = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, device=device)

    annotated_frame = results[0].plot(boxes=False, kpt_line=1, line_width=1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(
        annotated_frame,
        "FPS : " + str(int(fps)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLOv8 People Pose Estimation", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
