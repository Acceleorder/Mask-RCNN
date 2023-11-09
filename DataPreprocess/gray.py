import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img_path = 'Figure2.png'  # 替换为你的图像文件路径
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 高斯滤波
blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 使用5x5的内核进行高斯滤波，可以根据需要调整内核大小

# 计算灰度直方图
hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])

# 绘制灰度直方图
plt.figure(figsize=(8, 6))
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist)
plt.xlim([0, 256])
plt.grid(True)

# 显示原图和滤波后的图像
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(blurred, cmap='gray'), plt.title('Gaussian Blurred Image')

plt.show()
