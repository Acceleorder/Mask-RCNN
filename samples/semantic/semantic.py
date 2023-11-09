# -*- coding: utf-8 -*-

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
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
#ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num=0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("/home/yuchen/项目/Mask_RCNN_1/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = ["fault","boudinage","fold"]

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15  # background + 15 semantic

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()

class SemanticDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    
    def obj_index(self,image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['labelnames']
            del labels[0]
        return len(labels)

    
    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['labelnames']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_semantic(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("semantic", 1, "fault")
        self.add_class("semantic", 2, "boudinage")
        self.add_class("semantic", 3, "fold")
 
        for i in range(count):
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "resultyaml/" + filestr + ".yaml"
            print(dataset_root_path + "resultmask/" + filestr + ".png")
            cv_img = cv2.imread(dataset_root_path + "image/" + filestr + ".tif")

            self.add_image("semantic", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
#         count = 1  # number of object
#         img = Image.open(info['mask_path'])
#         num_obj=self.obj_index(image_id)
#         #num_obj = self.get_obj_index(img)
#         mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
#         mask = self.draw_mask(num_obj, mask, img,image_id)
#         occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
#         for i in range(count - 2, -1, -1):
#             mask[:, :, i] = mask[:, :, i] * occlusion

#             occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("fault") !=-1:
                labels_form.append("fault")
            if labels[i].find("boudinage") != -1:
                labels_form.append("boudinage")
            if labels[i].find("fold") != -1:
                labels_form.append("fold")

        
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        count = len(set(labels_form))  # number of object
        img = Image.open(info['mask_path'])
#         img=cv2.imread(info['mask_path'],1)
        num_obj=self.obj_index(image_id)
        #num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train_model():
    #基础设置
    dataset_root_path="/home//Mask_RCNN_1/semantic_train"
    img_floder = dataset_root_path + "image"
    mask_floder = dataset_root_path + "resultmask"
    #yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    count = len(imglist)

    #train与val数据集准备
    dataset_train = SemanticDataset()
    dataset_train.load_semantic(count, img_floder, mask_floder, imglist,dataset_root_path)
    dataset_train.prepare()

    #print("dataset_train-->",dataset_train._image_ids)

    dataset_val = SemanticDataset()
    dataset_val.load_semantic(count, img_floder, mask_floder, imglist,dataset_root_path)
    dataset_val.prepare()

    #print("dataset_val-->",dataset_val._image_ids)

    # Load and display random samples
    #image_ids = np.random.choice(dataset_train.image_ids, 4)
    #for image_id in image_ids:
    #    image = dataset_train.load_image(image_id)
    #    mask, class_ids = dataset_train.load_mask(image_id)
    #    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    config=ShapeConfig()
    config.display()
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')



    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=10,
                layers="all")
    endtime=time.time()
    
class SemanticConfig(ShapesConfig):
    GPU_COUNT=1
    IMAGES_PER_GPU=1
    
def predict():
    config=SemanticConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    model_path=model.find_list()
    
    assert model_path !="","Provide path to trained weights"
    print("loading weights from ",model_path)
    model.load_weights(model_path,by_name=True)
    
    class_names = ['BG','LD']
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
    
if __name__=="__main__":
    train_model()
