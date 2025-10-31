import cv2

car = cv2.CascadeClassifier("../cascades/cars.xml")

cap = cv2.VideoCapture("../videos/video_cars.mp4")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Haar Cascade Car Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
