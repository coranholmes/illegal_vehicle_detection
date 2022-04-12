# -*- coding: utf-8 -*-
# @Time: 17/9/18 2:06 PM
# @Author: Weiling
# @File: utils.py
# @Software: PyCharm


import numpy as np
import os,sys,json

DS_NAME = 'ISLab'  # current support 'ISLab' and 'xd_ds'
SUFFIX_LENGTH = 4  # the length of suffix (.mp4: length = 4)
MATCH_TEMPLATE_THRESHOLD = 0.85  # iou must be larger than this threshold to match the template (default = 0.7)
SEE_FRAMES_THRESHOLD = 3  # in case template match doesn't work well on some frames, this allows t frames of wrong template matching (default = 5)
MAP_REGION_THRESHOLD = 0.85  # iou must be larger than this threshold to be recognized as the same ROI (default = 0.7)
ILLEGAL_PARKED_THRESHOLD = 5  # if the vehicle parks more than t frames, it will be marked as illegal
RESET_THRESHOLD = 3  # in case yolo doesn't work well on some frames, the algo keeps the memory of the detection history, but if the object is not detected within t frames, the region will be reset (default = 20)
VEHICLES = ['car', 'bus', 'truck']
SAVE_IMAGE_RES = True  # whether to save image results
DRAW_ON_DETECTION_RESULTS = True  # whether to draw the detection results based on yolo detection
MATCH_TEMPLATE_ON_GREY = True  # whether to convert to grey scale when doing template matching

EVALUATION_IOU_THRESHOLD = 0
ILLEGAL_PARKING_MAX_RATIO = 0.3
NOT_IN_IOU_THRESHOLD = 0.3  # 判断两个box是否为同一个，最小的IOU

class Region(object):
    def __init__(self, id, box, type='unknown'):
        top, left, bottom, right = box
        self.id = id
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
        return "Region:: type:%s tracked: %d top:%d left:%d bottom:%d right:%d, parked: %d occluded: %d deleted: %d " % (
            self.type, self.tracked, self.top, self.left, self.bottom, self.right, self.parked_time, self.occluded_time,
            self.deleted_time,
        )

def make_dir(path):
    if not os.path.exists(path):
        print('Creating path {}'.format(path))
        os.mkdir(path)

def make_video_subdir(ds_root):
    capture_output_path = os.path.join(ds_root, 'capture')
    make_dir(capture_output_path)
    text_output_path = os.path.join(ds_root, 'label')
    make_dir(text_output_path)
    video_output_path = os.path.join(ds_root, 'output')
    make_dir(video_output_path)
    return capture_output_path, text_output_path, video_output_path

def get_mask_regions(mask_path, pic_name):
    print("Open mask file " + mask_path)

    mask_file = open(mask_path)
    mask_dict = json.load(mask_file)

    for i in range(len(mask_dict)):
        if mask_dict[i]["imageName"] == pic_name:
            masks = mask_dict[i]["Data"]
            mask_regions = []

            if len(masks) > 0:
                masks = masks["svgArr"]
                for poly in masks:
                    poly = poly["data"]
                    points = []
                    for p in poly:
                        points.append((p["x"], p["y"]))
                    mask_regions.append(points)
            return mask_regions
    print("Cannot find the mask regions!")
    sys.exit(-1)

def get_exp_paras():
    name = ""
    name = name + str(ILLEGAL_PARKING_MAX_RATIO) + "_" + str(EVALUATION_IOU_THRESHOLD) + "_" + str(NOT_IN_IOU_THRESHOLD) + "__MD_" + str(SEE_FRAMES_THRESHOLD) + "_" + str(RESET_THRESHOLD)
    return name