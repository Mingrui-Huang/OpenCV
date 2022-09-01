import numpy as np
import cv2
import os


def show_img(windowname, img):
    cv2.imshow(windowname, img)
    cv2.waitKey(10)


def makeVideo(video_name, src_path, size=(640, 480), fps=24):
    filelist = [os.path.join(src_path, i) for i in os.listdir(src_path)]
    fps = fps
    size = size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite = cv2.VideoWriter(video_name, fourcc, fps, size)
    for item in filelist:
        if item.endswith('.png'):
            img = cv2.imread(item)
            videoWrite.write(img)
    videoWrite.release()


# 读取和保存路径
src_root = './imgs'
save_root = './imgs_box'
if not os.path.exists(save_root):
    os.mkdir(save_root)

# 用于筛选矩形框
min_w = 38
min_h = 24

# ========================== 获取并显示16位背景图 ===================================================
background = np.zeros((480, 640))  # 创建背景
ref_images = []
for i in range(len(os.listdir(src_root))):  # 抽出背景图并求平均
    if i < 147:
        ref_image = cv2.imread('./imgs/{}.png'.format(i), cv2.CV_16UC1)
        ref_images.append(ref_image)
        background += ref_image / len(ref_images)

background = ((background - background.min()) / (background.max() - background.min())) * 255  # 归一化，便于观察
background = background.astype(np.uint8)

# ============================ 读取并显示16位背景与前景图 =====================================
imglist = os.listdir(src_root)
for index in range(len(imglist)):
    cur_img = cv2.imread(os.path.join(src_root, imglist[index]), cv2.CV_16UC1)
    cur_img = ((cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())) * 255
    cur_img = cur_img.astype(np.uint8)
    cv2.putText(cur_img, 'frame:' + str(index), (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # show_img('cur_img', cur_img)

    cur_img2 = cur_img.copy()  # 拷贝原图，用于对比
    # ======================================== 做帧差 ========================================
    frame_diff = cv2.absdiff(cur_img, background)
    # show_img('frame_diff', frame_diff)

    frame_diff_roi = frame_diff[-400:, -400:-100]  # 划出roi
    # show_img('frame_diff_roi', frame_diff_roi)

    frame_diff_roi_bin = cv2.inRange(frame_diff_roi, 10, 50)  # 二值化
    # show_img('frame_diff_roi_bin', frame_diff_roi_bin)

    blur = cv2.GaussianBlur(frame_diff_roi_bin, (3, 3), 0)  # 去噪
    # show_img('frame_diff_roi_bin', blur)

    # =======================================腐蚀膨胀========================================
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 获取卷积核
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilate = cv2.dilate(frame_diff_roi_bin, kernel, iterations=1)  # 膨胀

    erode = cv2.erode(dilate, kernel, iterations=1)  # 腐蚀
    # show_img('dilate', dilate)
    # show_img('erode', erode)

    # ==================查找轮廓==========================================
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        (x, y), radius = cv2.minEnclosingCircle(cont)  # 最小外接圆
        if 15 < radius < 40:  # 半径筛选
            x, y, w, h = cv2.boundingRect(cont)  # 最大外接矩形
            is_valid = (w >= min_w) and (h >= min_h)  # 筛选矩形框
            if not is_valid:
                continue
            x_new = x + (640 - 400)  # 根据roi范围确定矩形框在原图位置
            y_new = y + (480 - 400)

            cv2.rectangle(cur_img2, (x_new, y_new), (x_new + w, y_new + h), 255, 2)  # 画矩形框
    cv2.imshow('object with box', np.hstack((cur_img, cur_img2)))
    cv2.waitKey(10)
    cv2.imwrite(os.path.join(save_root, imglist[index]), cur_img2)
cv2.destroyAllWindows()

makeVideo('video.mp4', './imgs_box', size=(640, 480), fps=24)
