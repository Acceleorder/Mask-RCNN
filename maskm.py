# -*- coding: utf-8 -*-
"""
@author: liguo_yao
"""
 
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
 
 
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
 
mnist = read_data_sets("F:\\python\\CNN\\Mnist\\MNIST_data/", one_hot=True)
 
#mnist=read_data_sets('E:/mnist',one_hot=False)
def Image_Processing(image):
    '''缩放到32x32，像素值转成0、1'''
    image=np.reshape(image,[28,28]) # [28,28]
    img=cv2.resize(image,(32,32)) # [32,32]
 
 
    img_mask = np.round(img) # 转成对应的掩膜 像素值0、1
    return img,img_mask
 
 
def random_comb(mnist):
    num_images=mnist.train.num_examples
    
    list=range(0,num_images)
    indexs=random.sample(list,16) # 随机选择16个索引值
    
 
 
    indexs=np.asarray(indexs,np.uint8).reshape([4,4])
    class_ids = mnist.train.labels[indexs.flatten()]
    comb_image=np.zeros([32*4,32*4],np.float32)
    mask=[]
    for i in range(4):
        for j in range(4):
            image_mask = np.zeros([32 * 4, 32 * 4], np.uint8)
            img_data=mnist.train.images[indexs[i,j]]
            img, img_mask=Image_Processing(img_data)
            comb_image[i*32:(i+1)*32,j*32:(j+1)*32]=img
            image_mask[i*32:(i+1)*32,j*32:(j+1)*32]=img_mask
            mask.append(image_mask)
 
 
    return comb_image,np.asarray(mask).transpose([1,2,0]),class_ids
 
 
comb_image,mask,class_ids=random_comb(mnist)
 
 
print(class_ids)
 
 
plt.subplot(121)
plt.imshow(comb_image,'gray')
plt.title('original')
plt.axis('off')
 
 
plt.subplot(122)
plt.imshow(mask[:,:,9],'gray')
plt.title(class_ids[9])
plt.axis('off')
 
 
plt.show()