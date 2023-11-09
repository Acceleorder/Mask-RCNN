import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


img_path = r"Figure2.png"
img_bgr = cv2.imread(img_path, -1)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
fft = np.fft.fft2(img_gray)
fft_shift = np.fft.fftshift(fft)

abs_fft = np.abs(fft_shift)
# abs_fft = fft_shift
'''必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0'''
abs_fft = np.log(0.01+abs_fft)
fft_aug_norm = cv2.normalize(abs_fft, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

h,w = img_gray.shape
# 3D可视化，需要3维坐标，np.meshgrid将重复的复制简单化
x,y = np.meshgrid(np.arange(0,w), np.arange(0,h))
fig = plt.figure()
#ax1 = Axes3D(fig)
ax1 = plt.axes(projection='3d')
#plt.plot(fft_aug_norm, cmap='rainbow')
ax1.plot_surface(x, y, fft_aug_norm, cmap='rainbow')  # 这种颜色比较好一点
ax1.set_xlabel('X wave number',fontsize=12)
ax1.set_ylabel('Y wave number',fontsize=12)
ax1.set_zlabel('Frequency',fontsize=12)
# ax1.xaxis.get_label().set_fontsize(20)
plt.savefig('Figure1(f).jpg', format='jpg')
plt.show()