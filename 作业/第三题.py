import cv2
import numpy as np
import matplotlib.pyplot as plt


# 制作低通掩膜
def make_lowpass_mask(image, mask_cols=40, mask_rows=40):
    mask = np.zeros((len(image[1]), len(image[0])), dtype=int)
    center = (int(len(image[1]) / 2), int(len(image[0]) / 2))
    mask[
    center[0] - mask_cols // 2: center[0] + mask_cols // 2,
    center[1] - mask_rows // 2: center[1] + mask_rows // 2] = 1
    return mask


# 制作高通掩膜
def make_highpass_mask(image, mask_cols=40, mask_rows=40):
    mask = np.ones((len(image[1]), len(image[0])), dtype=int)
    center = (int(len(image[1]) / 2), int(len(image[0]) / 2))
    mask[
    center[0] - mask_cols // 2: center[0] + mask_cols // 2,
    center[1] - mask_rows // 2: center[1] + mask_rows // 2] = 0
    return mask


# fft，中心化
def fshift_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift


if __name__ == '__main__':
    # 读取lena图像，fft
    img = cv2.imread('lena.jpg', 0)
    img_fshift = fshift_fft(img)
    img_fshift_fft = np.log(np.abs(img_fshift))

    # 低通处理
    lowpass_mask = make_lowpass_mask(img, 50, 50)
    lowpass_fshift = lowpass_mask * img_fshift
    lowpass_fshift_fft = np.log(1 + np.abs(lowpass_fshift))
    img_lowpass = np.fft.ifftshift(lowpass_fshift)
    img_lowpass = np.fft.ifft2(img_lowpass)
    img_lowpass = np.abs(img_lowpass)

    # 高通处理
    highpass_mask = make_highpass_mask(img, 60, 60)
    highpass_fshift = highpass_mask * img_fshift
    highpass_fshift_fft = np.log(1 + np.abs(highpass_fshift))
    img_highpass = np.fft.ifftshift(highpass_fshift)
    img_highpass = np.fft.ifft2(img_highpass)
    img_highpass = np.abs(img_highpass)

    # 绘制
    plt.figure('work3', figsize=(5, 15))
    plt.subplot(321), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(322), plt.imshow(img_fshift_fft, 'gray'), plt.title('img_fft')
    plt.subplot(323), plt.imshow(img_lowpass, 'gray'), plt.title('lowpass_img')
    plt.subplot(324), plt.imshow(lowpass_fshift_fft, 'gray'), plt.title('lowpass_fft')
    plt.subplot(325), plt.imshow(img_highpass, 'gray'), plt.title('lowpass_img')
    plt.subplot(326), plt.imshow(highpass_fshift_fft, 'gray'), plt.title('lowpass_fft')

    plt.tight_layout()
    plt.show()
