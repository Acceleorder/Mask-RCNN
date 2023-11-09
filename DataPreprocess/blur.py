import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
img_path = 'Figure2.png'  # 替换为你的图像文件路径
image = cv2.imread(img_path)

# 高斯滤波
blurred = cv2.GaussianBlur(image, (11, 11), 0)  # 使用5x5的内核进行高斯滤波，可以根据需要调整内核大小

# 保存模糊处理后的图像
output_path = 'blurred_image.jpg'  # 替换为你想保存的文件路径及名称
cv2.imwrite(output_path, blurred)

# 计算直方图
hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])

# 显示原图和模糊处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)), plt.title('Blurred Image')

plt.show()
