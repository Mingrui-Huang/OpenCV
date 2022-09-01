import cv2
import numpy as np
import os

import handTrackingModule as htm

# 贴图路径
jpg_path = 'pen_and_eraser'
mylist = os.listdir(jpg_path)
overlayList = []
for imgPath in mylist:
    image = cv2.imread(os.path.join(jpg_path, imgPath))
    overlayList.append(image)

# 初始贴图， 颜色， 笔刷粗细，初始点, 笔刷样式, 延迟点，平滑因子
header = overlayList[0]
drawColor = (0, 0, 0)
brushThickness = 10
pp = (0, 0)
mode = 0

point_delay = (0, 0)
# 平滑因子越大月延迟， 月平滑
smoothening = 1

# 画布
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# 获取相机
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(mirror=True, detectionCon=0.85)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    img[0:120, 0:1280] = header
    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:

        p0 = lmList[4][1:]
        p1 = lmList[8][1:]
        p2 = lmList[12][1:]
        p3 = lmList[0][1:]
        p4 = lmList[9][1:]


        handtype = detector.handtype()
        fingers = detector.fingersUp()
        # print(handtype, fingers)
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            pp = (0, 0)
            print("selection mode")
            cx = (p1[0] + p2[0]) * 0.5
            cy = (p1[1] + p2[1]) * 0.5
            center = (int(cx), int(cy))
            cv2.line(img, p1, p2, (255, 0, 255), 2)
            # cv2.circle(img, center, 20, (255, 255, 255), 3)
            cv2.rectangle(img, (center[0]-17, center[1]-17), (center[0]+17, center[1]+17), (255, 255, 255), 2)
            cv2.rectangle(img, (center[0]-15, center[1]-15), (center[0]+15, center[1]+15), drawColor, cv2.FILLED)


            # 根据选中的图案选择笔刷
            if center[1] < 120:
                if 340 < center[0] < 460:
                    mode = 1
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 560 < center[0] < 640:
                    mode = 2
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 760 < center[0] < 860:
                    mode = 3
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 1000 < center[0] < 1100:
                    mode = 0
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

        # 亮出食指进行绘画
        if fingers[1] == 1 and fingers[2] == 0:
            if mode == 0:
                print("using eraser")
            elif mode == 1:
                print("using redPen")
            elif mode == 2:
                print("using greenPen")
            elif mode == 3:
                print("using blurPen")

            # 运动平滑处理
            p1[0] = int(
                point_delay[0] + (p1[0] - point_delay[0]) / smoothening)
            p1[1] = int(
                point_delay[1] + (p1[1] - point_delay[1]) / smoothening)

            cv2.circle(img, p1, brushThickness+1, (255, 255, 255), 3)
            cv2.circle(img, p1, brushThickness, drawColor, cv2.FILLED)
            cv2.circle(canvas, p1, brushThickness, drawColor, cv2.FILLED)
            point_delay = p1
            if pp == (0, 0):
                pp = p1
            cv2.line(canvas, pp, p1, drawColor, 2 * brushThickness)

            pp = p1
        # 快速橡皮差
        if fingers == [1, 1, 1, 1, 1]:
            print("using eraser")
            mode = 0
            cx2 = 0.5 * (p3[0] + p4[0])
            cy2 = 0.5 * (p3[1] + p4[1])

            center2 = (int(cx2), int(cy2))
            cv2.circle(img, center2, 100, (0, 0, 0), cv2.FILLED)
            cv2.circle(canvas, center2, 100, (0, 0, 0), cv2.FILLED)

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2RGB)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    # img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    cv2.imshow("image", img)
    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

