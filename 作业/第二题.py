import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.ndimage import rotate


import matplotlib.pyplot as plt

# 生成初始图像
f1 = np.zeros((128, 128), dtype=int)
for i in range(38, 90):
    for j in range(58, 70):
        f1[i, j] = 255

f2 = np.zeros((len(f1[0]), len(f1[1])), dtype=int)
for m in range(len(f1[0])):
    for n in range(len(f1[1])):
        f2[m, n] = pow(-1, (m + n)) * f1[m, n]

f3 = rotate(f2, angle=-45)

# f1进行fft
f1_fft = np.fft.fft2(f1)
shiftcenter1 = np.fft.fftshift(f1_fft)
log_fft1 = np.log(1 + np.abs(shiftcenter1))

# f2进行fft
f2_fft = np.fft.fft2(f2)
shiftcenter2 = np.fft.fftshift(f2_fft)
log_fft2 = np.log(1 + np.abs(shiftcenter2))

# f3进行fft
f3_fft = np.fft.fft2(f3)
shiftcenter3 = np.fft.fftshift(f3_fft)
log_fft3 = np.log(1 + np.abs(shiftcenter3))



def first_question():

    plt.figure('question1')
    plt.subplot(121)
    plt.title('f1')
    plt.imshow(f1, cmap='gray')

    plt.subplot(122)
    plt.title('f1_fft')
    plt.imshow(log_fft1, cmap='gray')

    plt.tight_layout()
    plt.show()

def second_question():

    plt.figure('question2')
    plt.subplot(221)
    plt.title('f1')
    plt.imshow(f1, cmap='gray')

    plt.subplot(222)
    plt.title('f1_fft')
    plt.imshow(log_fft1, cmap='gray')

    plt.subplot(223)
    plt.title('f2_fft')
    plt.imshow(log_fft2, cmap='gray')

    plt.tight_layout()
    plt.show()

def third_question():

    plt.figure('question3')
    plt.subplot(221)
    plt.title('f1')
    plt.imshow(f1, cmap='gray')

    plt.subplot(222)
    plt.title('f1_fft')
    plt.imshow(log_fft1, cmap='gray')

    plt.subplot(223)
    plt.title('f2_fft')
    plt.imshow(log_fft2, cmap='gray')

    plt.subplot(224)
    plt.title('f3_fft')
    plt.imshow(log_fft3, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    first_question()
    second_question()
    third_question()
