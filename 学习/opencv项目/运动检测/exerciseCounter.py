import cv2
import numpy as np
import time

import poseModule as pm


def saveExerciseVideo():
    save_path = './videos/exercise.mp4'
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, 24, (640, 480))
    while True:
        success, img = cap.read()
        video_writer.write(img)
        cv2.imshow('image', img)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()


is_right = True
# cap = cv2.VideoCapture('./videos/exercise.mp4')
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

count = 0
dir = 0

pTime = 0
while True:

    success, img = cap.read()
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        if is_right:
            if lmList[14][2] < lmList[12][2]:
                is_right = False
        if not is_right:
            if lmList[13][2] < lmList[11][2]:
                is_right = True
    print("right arm" if is_right else "left arm")
    if len(lmList) != 0:
        # right arm
        if is_right:
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (50, 150), (0, 100))
        # left arm
        else:
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (200, 310), (0, 100))
        print('angle: {},  percent: {}'.format(int(angle), int(per)))

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

    cv2.rectangle(img, (600, 300), (640, 400), (255, 255, 255), 2)

    if is_right:
        cv2.rectangle(img, (600, 300 + int(per)), (640, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "{}%".format(int(100 - per)), (580, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        cv2.rectangle(img, (600, 400 - int(per)), (640, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "{}%".format(int(per)), (580, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    if count - round(count) == 0:
        print(count)
        cv2.putText(img, str(count), (560, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv2.imshow('image', img)
    if cv2.waitKey(1000 // 24) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
