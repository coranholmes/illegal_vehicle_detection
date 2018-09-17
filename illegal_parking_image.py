# -*- coding: utf-8 -*-
# @Time: 10/9/18 3:28 PM
# @Author: Weiling
# @File: illegal_parking_image.py
# @Software: PyCharm


from yolo import YOLO, detect_video
from PIL import Image, ImageFont, ImageDraw
import os
from utils import *
import cv2
from matplotlib import pyplot as plt


test_path = os.path.join(os.getcwd(), 'images')
frame_cnt = 0
for fn in os.listdir(os.path.join(test_path, 'frames')):
    frame_cnt += 1

region_list = []
illegal_list = []

# vehicle detection for frame 0
yolo = YOLO()
frame_path = os.path.join(test_path, 'frames', 'ISLab-13-0.jpg')
image = Image.open(frame_path)
image_canvas, out_boxes, out_scores, out_classes = yolo.detect_image(image)
for i in range(len(out_boxes)):
    class_name = yolo.class_names[out_classes[i]]
    if class_name in VEHICLES:
        region = Region(out_boxes[i], class_name)
        region_list.append(region)

for idx in range(1, frame_cnt):
    print("Processing frame %d" % (idx * 25))
    prev_frame_path = os.path.join(test_path, 'frames', 'ISLab-13-' + str((idx - 1) * 25) + '.jpg')  # previous frame
    curr_frame_path = os.path.join(test_path, 'frames', 'ISLab-13-' + str(idx * 25) + '.jpg')  # current frame
    cvimage = np.asarray(image)  # image here is the previous image

    # template matching
    for r in region_list:
        if r.deleted_time > 0:
            continue
        else:
            template = cvimage[r.top:r.bottom, r.left:r.right]
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            img = cv2.imread(curr_frame_path, 0)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc

            top = max_loc[1]
            left = max_loc[0]
            bottom = max_loc[1] + h
            right = max_loc[0] + w

            matched_box = np.array([top, left, bottom, right], dtype=np.float32)
            matched_region = Region(matched_box)
            iou = r.get_iou(matched_region)

            if iou > MATCH_TEMPLATE_THRESHOLD:
                if r.tracked:
                    r.parked_time += 1
                else:
                    r.tracked = True
                    r.parked_time = r.parked_time + r.occluded_time + 1
                    r.occluded_time = 0
            else:
                if r.tracked:
                    r.occluded_time += 1
                    r.tracked = False
                else:
                    if r.occluded_time > SEE_FRAMES_THRESHOLD:
                        r.deleted_time += 1  # delete the vehicle
                    else:
                        r.occluded_time += 1

    # Look at region list and trigger alarm for those parked time longer than threshold
    for r in region_list:
        if r.parked_time > ILLEGAL_PARKED_THRESHOLD and r.deleted_time < 1:
            thickness = (image.size[0] + image.size[1]) // 300
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            label = "%ds,%d" % (r.parked_time, r.tracked)
            draw = ImageDraw.Draw(image_canvas)
            label_size = draw.textsize(label, font)

            box = r.get_box()
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='black')
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill='white')
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
    image_canvas.save(os.path.join(test_path, 'out', str((idx - 1) * 25) + '.jpg'), quality=90)

    # vehicle detection for current frame
    image = Image.open(curr_frame_path)
    image_canvas, out_boxes, out_scores, out_classes = yolo.detect_image(image)
    flag = []
    region_list_len = len(region_list)
    for i in range(len(out_boxes)):
        class_name = yolo.class_names[out_classes[i]]
        if class_name in VEHICLES:
            region = Region(out_boxes[i], class_name)
            # judge whether the detected region is already in two lists
            r_idx = region.find_region(region_list)
            if r_idx == -1:  # if cannot find the region in list than append it to the list
                region_list.append(region)
            else:
                flag.append(r_idx)
    # for those regions in the list who are not mapped to, r.traced = false, r.deleted_time += 1
    for i in range(region_list_len):
        if (region_list[i].deleted_time < 1):
            if (i not in flag):
                region_list[i].deleted_time += 1
                region_list[i].tracked = False
        else:
            region_list[i].deleted_time += 1
            region_list[i].tracked = False
            if region_list[i].deleted_time > RESET_THRESHOLD:
                region_list[i].parked_time = 0
                region_list[i].occluded_time = 0
                region_list[i].deleted_time = 0
                region_list[i].tracked = False
