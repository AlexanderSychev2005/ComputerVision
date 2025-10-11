import cv2
import numpy as np


def MeanShift(cap):
    timer = cv2.getTickCount()
    ret, frame = cap.read()

    x, y, w, h = cv2.selectROI(frame)
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        '''
        Apply the MeanShift algorithm
        '''
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw the track window on the frame
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        cv2.putText(img2, "FPS : " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('frame', img2)


        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


def CamShift(cap):
    ret, frame = cap.read()

    x, y, w, h = cv2.selectROI(frame)
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        '''
        Apply the CamShift algorithm
        '''

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw the track window on the frame
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.putText(img2, "FPS : " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


def CRFTrack(cap, tracker_type='CSRT'):
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()

    ret, frame = cap.read()

    x, y, w, h = cv2.selectROI(frame)
    track_window = (x, y, w, h)

    ret = tracker.init(frame, track_window)
    while True:
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        ret, track_window = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if ret:
            x, y, w, h = map(int, track_window)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    cap = cv2.VideoCapture("ballet.mp4")
    # MeanShift(cap)
    # CamShift(cap)
    CRFTrack(cap, tracker_type='CSRT')
    # CRFTrack(cap, tracker_type='KCF')
    # CRFTrack(cap, tracker_type='MOSSE')
