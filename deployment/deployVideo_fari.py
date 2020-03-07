#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
import keras

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)

# ## Load RetinaNet model
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
# model_path = os.path.join('..', 'snapshots', 'resnet101_csv_03.h5')
# model_path = os.path.join('..', 'snapshots', 'resnet101_csv_30.h5')
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
#model_path = os.path.join('..', 'snapshots', 'vis-resnet50/resnet50_csv_97.h5')
# model_path = os.path.join('..', 'snapshots', 'hardHat-resnet50', 'resnet50_csv_21.h5')

# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet101')
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes

# COCO class labels

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                   8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                   14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                   22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
                   29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                   35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                   40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                   47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                   54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                   61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                   68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                   75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


#labels_to_names = {0: 'car', 1: 'truck', 2: 'van', 3: 'bus'}
"""
labels_to_names = {0: 'helmet', 1: 'no-helmet'}
"""
# ## Run detection on example
# load image
# name = "04055001"
# name = '02315001'
# name = '02315002'
# name = '02316001'
# dirname = './img/'
dirname = '../tests/test-data/Coco/images/val2017/'

count = 0
xmlList = []

for root, dirs, files in os.walk(dirname):
    for f in files:
        if count > 1000:
            break

        image=cv2.imread(root+'/'+f)
        img_array = []
        height, width, layers = image.shape
        size = (width, height)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # print("processing time: ", (time.clock() - start), "s")

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if label == 0:
            # scores are sorted so we can break
                if score < 0.66:
                    continue

                count += 1
                value = (f, int(box[0]), int(box[1]), int(box[2]), int(box[3]), 'vest')
                xmlList.append(value)

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                # caption = ""
                draw_caption(draw, b, caption)


                cv2.imwrite("./outputTEST/Frame%d.jpg" % count, draw)
                #
                # img_array.append(draw)

xmlDF = pd.DataFrame(xmlList)
xmlDF.to_csv("./COCOpeopleAnnotation.csv", index=None, header=False)
