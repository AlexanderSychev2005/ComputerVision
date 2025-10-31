from collections import defaultdict


import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8m.pt")


INDEXES = [
    0,
    1,
    2,
    3,
    5,
    7,
]  # COCO class IDs for person, bicycle, car, motorcycle, bus, truck
CONFIDENCE = 0.4

track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture("../videos/video_cars.mp4")


while cap.isOpened():

    timer = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        device=device,
        persist=True,
        verbose=False,
        conf=CONFIDENCE,
        classes=INDEXES,
    )

    annotated_frame = results[0].plot(line_width=1)
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xywh.cpu()  # (x1, y1, x2, y2)
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 10:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=4,
            )
            if len(track) > 5:
                current_pos = track[-1]
                old_pos = track[-5]

                vx = (current_pos[0] - old_pos[0]) / 5
                vy = (current_pos[1] - old_pos[1]) / 5

                future_pos_x = int(current_pos[0] + vx * 10)
                future_pos_y = int(current_pos[1] + vy * 10)

                cv2.arrowedLine(
                    annotated_frame,
                    (int(current_pos[0]), int(current_pos[1])),
                    (future_pos_x, future_pos_y),
                    (0, 255, 0),
                    thickness=2,
                )

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

    cv2.imshow("YOLOv8 Car Detection and Tracking", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
