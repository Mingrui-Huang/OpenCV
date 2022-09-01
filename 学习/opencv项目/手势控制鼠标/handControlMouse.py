import cv2
import numpy as np
import time
import autopy
import handTrackingModule as htm

wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 100  # frame reduction
smoothening = 3

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, mirror=True)

pTime = 0
prior_X, prior_Y = 0, 0
current_X, current_Y = 0, 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 找到landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    # 找到食指和中值尖
    if len(lmList) != 0:

        p0 = lmList[0][1:]
        p1 = lmList[8][1:]
        p2 = lmList[12][1:]

        fingers = detector.fingersUp()
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255))
            cv2.circle(img, p1, 10, (255, 0, 255), cv2.FILLED)
            x = np.interp(p1[0], (frameR, wCam - frameR), (0, wScr))
            y = np.interp(p1[1], (frameR, hCam - frameR), (0, hScr))

            # 无阻尼， 抖动
            # autopy.mouse.move(x, y)
            # 有阻尼，顺滑
            current_X = prior_X + (x - prior_X)/smoothening
            current_Y = prior_Y + (y - prior_Y)/smoothening
            autopy.mouse.move(current_X, current_Y)
            prior_X, prior_Y = current_X, current_Y

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            length, img, center = detector.findDistance(8, 12, img)
            if length < 30:
                cv2.circle(img, p1, 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click(autopy.mouse.Button.LEFT, 1)

    # 帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # 显示
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break
