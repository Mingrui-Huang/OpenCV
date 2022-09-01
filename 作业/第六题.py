import cv2
import numpy as np
import matplotlib.pyplot as plt


def roberts(img):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

def prewitt(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel


def laplacian(img):
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian


def canny(img):
    Canny = cv2.Canny(img, 50, 150)
    return Canny

def show(fname):
    if fname != 'road-SAR.png':
        img = cv2.imread(fname, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(fname, cv2.COLOR_BGR2GRAY)

    img_roberts = roberts(img)
    img_prewitt = prewitt(img)
    img_sobel = sobel(img)
    img_laplacian = laplacian(img)
    img_canny = canny(img)

    plt.figure(num= fname)
    plt.subplot(231)
    plt.title('original')
    plt.imshow(img, 'gray')

    plt.subplot(232)
    plt.title('roberts')
    plt.imshow(img_roberts, 'gray')

    plt.subplot(233)
    plt.title('prewitt')
    plt.imshow(img_prewitt, 'gray')

    plt.subplot(234)
    plt.title('sobel')
    plt.imshow(img_sobel, 'gray')

    plt.subplot(235)
    plt.title('laplacian')
    plt.imshow(img_laplacian, 'gray')

    plt.subplot(236)
    plt.title('canny')
    plt.imshow(img_canny, 'gray')
    plt.show()

if __name__ == '__main__':
    show('iris-Na.tif')
    show('bridge-RS.jpg')
    show('road-SAR.png')

