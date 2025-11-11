import os
import time
from ultralytics import YOLO
import cv2
import torch
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Yolo configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = YOLO("yolov8n-face.pt")

# DeepFace configuration
threshold = 0.4
owner_folder = "./faces"
model_name = 'VGG-Face'

face_vectors = []
if not os.path.exists(owner_folder):
    print("The owner face folder does not exist. Please create the folder and add owner's face images.")
    exit()

print(f"Owner's face folder: {owner_folder}")
for filename in os.listdir(owner_folder):
    if filename.endswith(('jpg', '.png', 'jpeg')):
        filepath = os.path.join(owner_folder, filename)

        try:
            representation = DeepFace.represent(img_path=filepath,
                                     model_name=model_name,
                                     enforce_detection=True
                                     )
            face_vector = representation[0]["embedding"]
            face_vectors.append(face_vector)
            print(f"Loaded and processed owner's face image: {filename}")
        except Exception as e:
            print(f"Could not process image {filename}: {e}")

if not face_vectors:
    print("No valid owner's face images found. Exiting.")
    exit()

print(f"Owner's face images processed successfully. There are total {len(face_vectors)} images.")

cap = cv2.VideoCapture(2)

last_check_time = 0
check_interval = 2
last_seen_person = "Unknown"
last_distance = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    if current_time - last_check_time > check_interval:
        print("Checking...")
        last_check_time = current_time
        last_seen_person = "Unknown"

        results = model(frame, device=device, verbose=False)
        for result in results:
            if len(result.boxes.xyxy):
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    try:
                        padding = 25
                        face_crop = frame[max(0, y1 - padding):min(frame.shape[0], y2 + padding),
                                            max(0, x1 - padding):min(frame.shape[1], x2 +padding)]

                        # verification = DeepFace.verify(
                        #     img1_path=face_crop,
                        #
                        #     img2_path=owner_image_path,
                        #     enforce_detection=False,
                        #     distance_metric="cosine",
                        #     model_name='VGG-Face'
                        # )
                        # if verification["verified"]:
                        #     last_seen_person = "Owner"
                        #     print("Owner detected.")
                        #     break
                        live_representation = DeepFace.represent(img_path=face_crop,
                                                                 model_name=model_name,
                                                                 enforce_detection=False)
                        live_vector = live_representation[0]["embedding"]

                        isOwner = False
                        min_distance = 1.0 # 1 - 90 degrees, no similarity, 0 - 0 degrees, similarity


                        for owner_vector in face_vectors:
                            distance = cosine(owner_vector, live_vector)
                            if distance < min_distance:
                                min_distance = distance


                            if distance < threshold:
                                isOwner = True
                                break
                        last_distance = min_distance

                        if isOwner:
                            last_seen_person = "Owner"
                            print("Owner detected.")
                            break

                    except Exception as e:
                        pass
            if last_seen_person == "Owner":
                break
        if last_seen_person == "Unknown":
            print("Unknown person detected.")

    if last_seen_person == "Owner":
        colour = (0, 255, 0)
        text = f"Owner"
    else:
        colour = (0, 0, 255)
        text = f"Unknown"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    if last_distance > 0.0:
        cv2.putText(frame, f"Min Dist: {last_distance:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)



    cv2.imshow("Smart Lock System (YOLO + DeepFace)", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()