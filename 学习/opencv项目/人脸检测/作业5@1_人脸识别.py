import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cv2.namedWindow('get_video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('get_video', 640, 480)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(frameRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = [int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)]
            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, str(round(detection.score[0], 2)), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 255), 2)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow('get_video', frame)
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()
