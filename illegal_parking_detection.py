# -*- coding: utf-8 -*-
# @Time: 10/9/18 3:28 PM
# @Author: Weiling
# @File: illegal_parking_detection.py
# @Software: PyCharm


from yolo import YOLO, detect_video
from PIL import Image
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

test_path = os.getcwd()
idx = 0
image_file = 'images/frames/ISLab-13-' + str(idx) + '.jpg'
image = Image.open(os.path.join(test_path, image_file))
yolo = YOLO()
out_boxes, out_scores, out_classes = yolo.detect_image(image, True)

cvimage = cv2.imread(os.path.join(test_path, image_file))  
for i in range(len(out_boxes)):
    if yolo.class_names[out_classes[i]] == 'car':
        top, left, bottom, right = out_boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        cropped = cvimage[top:bottom, left:right]
        new_image_name = os.path.join(test_path, 'images', str(i) + ' ' + str(top) + ',' + str(bottom) + ';' + str(left) + ',' + str(right) + '.jpg')
        # cv2.imshow(new_image_name, cropped)
        cv2.imwrite(new_image_name, cropped)

