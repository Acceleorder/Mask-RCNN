
# coding: utf-8

# In[46]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN|
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/semantic/"))  # To find local version
import semantic

#get_ipython().magic('matplotlib inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
GIS_MODEL_PATH = os.path.join('/home/Mask_RCNN/', "trained_model.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(GIS_MODEL_PATH):
#     utils.download_trained_weights(GIS_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test_1")


# In[47]:


class InferenceConfig(semantic.ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# In[48]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(GIS_MODEL_PATH, by_name=True)


# In[49]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['fault','boudinage','fold'];


# In[50]:


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print(r['class_ids'])
print(r['class_ids'][0])
print(class_names)
#visualize.display_top_masks(image, r['rois'], r['masks'], r['class_ids'], class_names)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])


# In[ ]:




