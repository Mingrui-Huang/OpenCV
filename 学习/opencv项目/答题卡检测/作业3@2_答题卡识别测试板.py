import cv2
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
image = cv2.imread('./images/hbz.jpg')

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
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
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
# 显示变换结果
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 透视变换
warped_original = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped_gray = cv2.cvtColor(warped_original, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imwrite("./images/warped.jpg", thresh)
# 变换
orig_resize = resize(orig, height=800)
warped_resize = resize(warped_original, height=800)
thresh_resize = resize(thresh, height=800)
cv2.imshow("original", orig_resize)
cv2.imshow("Scanned", warped_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 找到选择题在的答题区
region_edges = cv2.Canny(thresh_resize, 75, 200)
kernel = np.ones((3, 3), dtype=np.uint8)
region_edges = cv2.dilate(region_edges, kernel=kernel, iterations=1)
region_edges = cv2.erode(region_edges, kernel=kernel, iterations=1)
cv2.imshow("region_edges", region_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 轮廓检测
contours = cv2.findContours(region_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

thresh_resize_BGR = cv2.cvtColor(thresh_resize, cv2.COLOR_GRAY2RGB)
answerBoxs = []
for contour in contours:
    # 轮廓近似
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon=0.02*peri, closed=True)
    # 4个点的时候拿出来
    if len(approx) == 4:
        answerBox = approx
        x, y, w, h = cv2.boundingRect(answerBox)
        if 2.8 < (w/h) < 3.2:
            answerBoxs.append(answerBox)

# print(len(answerBoxs))
# 在变换的图像上框出选择题区域
warped_resize_box = warped_resize.copy()
cv2.drawContours(warped_resize_box, [answerBoxs[0]], -1, (0, 255, 0), 2)
cv2.imwrite('./images/answer_zone_test.jpg', warped_resize_box)
cv2.imshow("region_edges", warped_resize_box)
cv2.waitKey(0)
cv2.destroyAllWindows()


answer_region = four_point_transform(warped_resize.copy(), answerBoxs[0].reshape(4, 2))
answer_region = cv2.resize(answer_region, (520, 170), interpolation=cv2.INTER_AREA)
answer_region = cv2.cvtColor(answer_region, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(answer_region, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imwrite('./images/thresh_test.jpg', thresh)
cv2.imshow("region_edges", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = cv2.erode(thresh, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)
cv2.imshow("answer_region", np.hstack((thresh, mask)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cavas = np.zeros(mask.shape[:2], dtype=np.uint8)
cavas2 = cavas.copy()
contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)
answer_centers = []
for contour in contours:
    # 轮廓近似
    x, y, w, h = cv2.boundingRect(contour)
    if w*h > 40 and 0.8 < (w/h) < 2.5:
        cv2.rectangle(cavas, (x, y), (x + w, y + h), 255, 1)
        answer_center = (y + h/2, x + w/2)
        answer_centers.append(answer_center)
    cv2.imshow('contours', np.hstack((mask, cavas)))
    cv2.waitKey(10)
cv2.waitKey(0)
cv2.destroyAllWindows()

correct_centers = np.load('./answer_centers.npy')
print(correct_centers)
score = 0
for answer_center in answer_centers:
    for correct_center in correct_centers:
        dis = np.sqrt(((answer_center[0] - correct_center[0])**2) + (answer_center[1] - correct_center[1])**2)
        if dis < 3:
            score = score + 1
print("score:{}" .format(score))

