import cv2
import argparse
import numpy as np


def order_points(pts):
    # 4个坐标点
    rect = np.zeros((4, 2), dtype=np.float32)
    # 按顺序0123分别为左上，右上，右下，左下
    # 计算左上， 右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右下， 左上
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # 获取输入的坐标值
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的高和宽
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    # 变换后的目标位置
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1,  maxHeight - 1],
                    [0, maxHeight - 1]], dtype=np.float32)
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(width)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 输入图像
image = cv2.imread('./images/cart.jpg')

# 坐标的变化
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# 预处理结果
print("STEP1: 边沿检测")
cv2.imshow('image', image)
cv2.imshow('edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 轮廓检测
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    # 轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon=0.02*peri, closed=True)
    # 4个点的时候拿出来
    if len(approx) == 4:
        screenCnt = approx
        break
# 显示结果
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped_gray, 160, 255, cv2.THRESH_BINARY)[1]
# cv2.imwrite("./images/warped.jpg", ref)
# 变换
cv2.imshow("original", resize(orig, height=800))
cv2.imshow("Scanned", resize(warped, height=800))
cv2.waitKey(0)
cv2.destroyAllWindows()