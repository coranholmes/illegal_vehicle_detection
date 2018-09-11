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

cvimage = np.asarray(image)
for i in range(len(out_boxes)):
    if yolo.class_names[out_classes[i]] == 'car':
        top, left, bottom, right = out_boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        template = cvimage[top:bottom, left:right]
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # new_image_name = os.path.join(test_path, 'images', str(i) + ' ' + str(top) + ',' + str(bottom) + ';' + str(left) + ',' + str(right) + '.jpg')
        new_image_name = os.path.join(test_path, 'images', str(i) + '.jpg')
        # cv2.imwrite(new_image_name, template)
        img = cv2.imread('images/frames/ISLab-13-25.jpg', 0)
        img2 = img.copy()
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print(i)
        print(left, top, right, bottom)
        print(top_left, bottom_right)
