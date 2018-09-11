# -*- coding: utf-8 -*-
# @Time: 11/9/18 10:21 AM
# @Author: Weiling
# @File: split_frames.py
# @Software: PyCharm


import cv2

i = '01'
vidcap = cv2.VideoCapture('videos/ISLab/ISLab-' + str(i) + '.mp4')
success, image = vidcap.read()
count = 0
while success:
    if count % 100 == 0:
        img_path = 'videos/ISLab_capture/' + str(i) +'/ISLab-' + str(i) + '-' + str(count) + '.jpg'
        cv2.imwrite(img_path, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
