import cv2
import matplotlib.pyplot as plt

# 读取图像
img_path = 'Figure2.png'  # 替换为你的图像文件路径
image = cv2.imread(img_path)

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 边缘检测
edges = cv2.Canny(gray_image, 100, 200)  # 可以根据需求调整阈值

# 显示原图和 Canny 边缘检测结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge Detection')

plt.show()
