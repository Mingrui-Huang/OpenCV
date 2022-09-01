import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_grayHist(grayHist,title):
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])
    plt.title(title)
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")


def gray_l8(img):
    gray_list = [0, 0, 0, 0.15, 0.35, 0.65, 0.85, 1]
    layer_length = 256 // 8
    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j] <= layer_length:
                img[i][j] = 255 * gray_list[0]
            elif img[i][j] <= 2*layer_length:
                img[i][j] = 255 * gray_list[1]
            elif img[i][j] <= 3*layer_length:
                img[i][j] = 255 * gray_list[2]
            elif img[i][j] <= 4*layer_length:
                img[i][j] = 255 * gray_list[3]
            elif img[i][j] <= 5*layer_length:
                img[i][j] = 255 * gray_list[4]
            elif img[i][j] <= 6*layer_length:
                img[i][j] = 255 * gray_list[5]
            elif img[i][j] <= 7*layer_length:
                img[i][j] = 255 * gray_list[6]
            elif img[i][j] <= 8*layer_length:
                img[i][j] = 255 * gray_list[7]
    return img


def gaussian_noise(img, mean, sigma):
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值

def blur(img):
    h, w = img.shape[0], img.shape[1]
    for i in range(1, h-1, 1):
        for j in range(1, w-1, 1):
            img[i][j] = (int(img[i][j-1]) + int(img[i][j+1]) + int(img[i-1][j]) + int(img[i+1][j])) // 4
    return img


img1 = cv2.imread('./pollen.tif')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', img1)
cv2.waitKey()
grayHist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])

# 直方图均衡化
img2 = cv2.equalizeHist(img1)
cv2.imshow('equalizeHist', img2)
cv2.waitKey()
grayHist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

# L8灰度级
img3 = gray_l8(img2)
cv2.imshow('gray_L8', img3)
cv2.waitKey()
grayHist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

# 添加高斯噪声
img_noise = gaussian_noise(img2, 0, 0.2)
cv2.imshow('gaussian_noise', img_noise)
cv2.waitKey()

# 4邻域平均法
img_4blur = blur(img_noise)
cv2.imshow('4-blur', img_4blur)
cv2.waitKey()

# 中值滤波
img_medianblur = cv2.medianBlur(img_noise, 3)
cv2.imshow('medianblur', img_medianblur)
cv2.waitKey()

plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_grayHist(grayHist1, 'original')
plt.subplot(132)
plot_grayHist(grayHist2, 'equalizeHist')
plt.subplot(133)
plot_grayHist(grayHist3, 'gray_L8')
plt.tight_layout()
plt.show()
