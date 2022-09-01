import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian_sharpe(img, kernel_size=3):
    h, w = img.shape[0], img.shape[1]
    pad = kernel_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float)
    out[pad: h + pad, pad: w + pad] = img.copy().astype(np.float)
    tmp = out.copy()

    K = np.array([
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]])

    for i in range(h):
        for j in range(w):
            out[pad + i, pad + j] = (-1) * np.sum(K * (tmp[i:i + kernel_size, j:j + kernel_size])) + tmp[
                pad + i, pad + j]
    out = np.clip(out, 0, 255)
    out = out[pad: pad + h, pad: pad + w].astype(np.uint8)
    return out


img = cv2.imread('./lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_sharpe = laplacian_sharpe(img, 3)

images = np.hstack([img, img_sharpe])
cv2.imshow('original and laplacian_sharpe', images)
cv2.waitKey()

