# -*- coding: utf-8 -*-
# @Time: 17/9/18 1:53 PM
# @Author: Weiling
# @File: illegal_parking_video.py
# @Software: PyCharm


import cv2
import os

from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
from yolo import YOLO, detect_video
from utils import *

video_path = os.path.join(os.getcwd(), 'videos', 'ISLab', 'ISLab-13.mp4')
output_path = os.path.join(os.getcwd(), 'videos', 'output', 'ISLab-13.mp4')

vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
# video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))


out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()

yolo = YOLO()
region_list = []
illegal_list = []

# vehicle detection for frame 0
return_value, cur_img_cv = vid.read()
cur_img_cv = cv2.cvtColor(cur_img_cv, cv2.COLOR_BGR2RGB)
image = Image.fromarray(cur_img_cv)  # transfer OpenCV format to PIL.Image format
image_canvas, out_boxes, out_scores, out_classes = yolo.detect_image(image)
# image_canvas.save(os.path.join(os.getcwd(),'images', 'out2', '0.jpg'), quality=90)

for i in range(len(out_boxes)):
    class_name = yolo.class_names[out_classes[i]]
    if class_name in VEHICLES:
        region = Region(out_boxes[i], class_name)
        region_list.append(region)

idx = 1
result = None

while True:
    pre_img_cv = np.asarray(image)  # transfer PIL.Image format to OpenCV format
    return_value, cur_img_cv = vid.read()

    if not return_value:
        break

    # template matching
    if idx % DETECT_EVERY_N_FRAMES == 0:
        print("Current time: %d" % idx)
        for r in region_list:
            if r.deleted_time > 0:
                continue
            else:
                template = pre_img_cv[r.top:r.bottom, r.left:r.right]
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # convert to grey scale

                w, h = template.shape[::-1]
                res = cv2.matchTemplate(cv2.cvtColor(cur_img_cv, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCORR_NORMED)
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
        if SAVE_IMAGE_RES:
            image_canvas.save(os.path.join(os.getcwd(),'images', 'out2', str(idx) + '.jpg'), quality=90)

        result = cv2.cvtColor(np.asarray(image_canvas), cv2.COLOR_RGB2BGR)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

    # if idx % DETECT_EVERY_N_FRAMES == 0:
        # vehicle detection for current frame
        cur_img_cv = cv2.cvtColor(cur_img_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cur_img_cv)  # transfer OpenCV format to PIL.Image format
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

    if result is None:
        out.write(cv2.cvtColor(np.asarray(image_canvas), cv2.COLOR_RGB2BGR))
    else:
        out.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    idx += 1
yolo.close_session()
