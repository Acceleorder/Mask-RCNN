import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
img_path = 'gray.png'  # 替换为你的图像文件路径
image = cv2.imread(img_path)

# 拆分图像为各通道
b, g, r = cv2.split(image)

# 计算各通道的直方图
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# 绘制RGB直方图
plt.figure(figsize=(10, 6))
plt.rc('font',size=15)
# plt.title('RGB Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist_r, color='red')
plt.rc('font',size=15)
plt.xlim([0, 256])
# plt.legend()
plt.grid(True)

# 保存直方图为JPG图像文件
plt.savefig('Figure6(d).jpg', format='jpg')

# 显示原图
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.show()
