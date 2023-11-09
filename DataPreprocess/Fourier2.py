import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img_path = 'Figure2.png'  # 替换为你的图像文件路径
image = cv2.imread(img_path, 0)  # 以灰度模式读取图像

# 进行二维傅里叶变换
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# 显示原始图像和频谱图（以灰度显示）
plt.figure(figsize=(12, 6))

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')

plt.show()