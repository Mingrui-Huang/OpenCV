import math
import cv2
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import handTrackingModule as htm


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.7, mirror=True)


pTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    cv2.rectangle(img, (20, 300), (60, 400), (255, 255, 255), 1)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])
        # 大拇指和食指尖
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # 手掌边
        x3, y3 = lmList[0][1], lmList[0][2]
        x4, y4 = lmList[17][1], lmList[17][2]

        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        # 计算两手指距离和掌边距离的比例
        length1 = math.hypot(x2 - x1, y2 - y1)
        length2 = math.hypot(x4 - x3, y4 - y3)
        scale = round(length1/length2, 2)
        # print(scale)

        vol = np.interp(scale, [0.3, 1.8], [minVol, maxVol])
        # print(vol, maxVol, minVol)
        volBar = 100 * (vol + maxVol) / (maxVol - minVol) + 100
        print(int(volBar))
        volume.SetMasterVolumeLevel(vol, None)
        if length1 < 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (20, 400 - int(volBar)), (60, 400), (0, 255, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps:" + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
