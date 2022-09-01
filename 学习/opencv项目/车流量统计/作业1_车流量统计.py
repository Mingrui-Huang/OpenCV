import cv2
import numpy as np

cap = cv2.VideoCapture('./videos/video.mp4')
# cap = cv2.VideoCapture(0)
# 创建背景对象
bgs = cv2.createBackgroundSubtractorMOG2()
# 创建腐蚀卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 设置过滤小框的阈值
min_w = 100
min_h = 90
# 设置线
line_height = 620
# 计算矩形中心点
def center(x, y, w, h):
    center_x = x + int(w/2)
    center_y = y + int(h/2)
    return center_x, center_y
# 中心点偏移量
offset = 6
cars = []
cars_num = 0

while True:
    ret, frame_org = cap.read()
    if ret == True:
        # 原始帧灰度化，去噪
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
        # 去噪
        blur = cv2.GaussianBlur(frame, (3, 3), 5)
        fgmask = bgs.apply(blur)
        # 腐蚀
        erode = cv2.erode(fgmask, kernel)
        # 膨胀
        dilate = cv2.dilate(erode, kernel, iterations=2)
        # 闭运算，消除内部噪声
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 画检测线
        cv2.line(frame_org, (10, line_height), (1200, line_height), (0, 255, 0), 2)
        # 画出轮廓
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            is_valid = (w >= min_w) and (h >= min_h)
            if not is_valid:
                continue
            cv2.rectangle(frame_org, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # 计算外接矩形的中心点
            center_point = center(x, y, w, h)
            cars.append(center_point)
            cv2.circle(frame_org, (center_point), 3, (0, 0, 255), 3)
            # 判断汽车是否过线
            for (x, y) in cars:
                if y > (line_height - offset) and y < (line_height + offset):

                    cars_num += 1
                    cars.remove((x, y))
                    print(cars_num)

        # 添加统计信息到frame
        cv2.putText(frame_org, 'vehicle count:' + str(cars_num), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow('video', frame_org)


    key = cv2.waitKey(30)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
