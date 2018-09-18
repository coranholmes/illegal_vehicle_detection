# -*- coding: utf-8 -*-
# @Time: 17/9/18 2:06 PM
# @Author: Weiling
# @File: utils.py
# @Software: PyCharm


import numpy as np

MATCH_TEMPLATE_THRESHOLD = 0.7  # iou must be larger than this threshold to match the template
SEE_FRAMES_THRESHOLD = 5  # in case template match doesn't work well on some frames, this allows t frames of wrong template matching
MAP_REGION_THRESHOLD = 0.7  # iou must be larger than this threshold to be recognized as the same ROI
ILLEGAL_PARKED_THRESHOLD = 5  # if the vehicle parks more than t frames, it will be marked as illegal
RESET_THRESHOLD = 20  # in case yolo doesn't work well on some frames, the algo keeps the memory of the detection history, but if the object is not detected within t frames, the region will be reset
VEHICLES = ['car', 'bicycle', 'motorbike', 'bus', 'truck']
SAVE_IMAGE_RES = True  # whether to save image results


class Region(object):
    def __init__(self, box, type='unknown'):
        top, left, bottom, right = box
        self.top = max(0, np.floor(top).astype('int32'))
        self.left = max(0, np.floor(left).astype('int32'))
        self.bottom = np.floor(bottom).astype('int32')
        self.right = np.floor(right).astype('int32')

        self.tracked = True
        self.parked_time = 0
        self.occluded_time = 0
        self.deleted_time = 0
        self.type = type

    def get_box(self):
        box = np.array([self.top, self.left, self.bottom, self.right], dtype=np.float32)
        return box

    def get_iou(self, boxB):
        boxA = self.get_box()
        boxB = boxB.get_box()

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def find_region(self, region_list):
        max_iou = 0
        max_idx = 0
        for i in range(len(region_list)):
            iou = self.get_iou(region_list[i])
            if (iou > max_iou) and (self.type == region_list[i].type):  # find the max iou of that type
                max_iou = iou
                max_idx = i
        if max_iou > MAP_REGION_THRESHOLD:
            if region_list[max_idx].deleted_time > 1:
                region_list[max_idx].parked_time += region_list[max_idx].occluded_time + region_list[
                    max_idx].deleted_time
                region_list[max_idx].occluded_time = 0
                region_list[max_idx].deleted_time = 0
                region_list[max_idx].tracked = True
            return max_idx
        else:
            return -1

    def __str__(self):
        return "Region:: top:%d left:%d bottom:%d right:%d, parked: %d occluded: %d tracked: %d" % (
            self.top, self.left, self.bottom, self.right, self.parked_time, self.occluded_time, self.tracked)
