import os, json
from evaluator import get_iou
from utils import *
from shapely.geometry import Polygon


def not_in(reg, reg_lst):
    for r in reg_lst:
       iou = get_iou(r[1], reg)
       if iou > 0:
           return False
    return True

if __name__ == "__main__":
    ds_root = os.path.join(os.getcwd(), "videos", "xd_full")
    label_path = os.path.join(ds_root, "label", "cloudy.txt")
    label_file = open(label_path)
    mask_regions = get_mask_regions(os.path.join(ds_root, "mask", "mask.json"), "cloudy.jpg")
    decs = dict()
    for line in label_file:
        # {"frame": 210, "id": 6, "type": "bus", "top": 212, "left": 117, "bottom": 343, "right": 286, "parked_time": 5, "detected": "YES"}
        d = json.loads(line)
        if d["detected"] == "YES":  # TODO positives做预处理删除特别小的框
            # dec_ids_set.add(d["id"])
            if d["frame"] not in decs:
                decs[d["frame"]] = []
            decs[d["frame"]].append(
                [
                    d["top"],
                    d["left"],
                    d["bottom"],
                    d["right"],
                ]
            )  
    dec_regs = []
    for frame in sorted(decs):
            for dec_id in range(len(decs[frame])):
                if not_in(decs[frame][dec_id], dec_regs):
                    dec_regs.append((frame, decs[frame][dec_id]))
    for frame, box in dec_regs:
        x1, y1, x2, y2 = box[1], box[0], box[3], box[2]  # TODO 检查
        poly1 = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])  # dtected bbox
        for poly2 in mask_regions:
            poly2 = Polygon(poly2)
            intersection_area = poly1.intersection(poly2).area
            a = poly1.area
            if intersection_area / a <= ILLEGAL_PARKING_MAX_RATIO:
                in_illegal_area = False
            else:
                in_illegal_area = True

        print(frame, box, intersection_area / a, in_illegal_area)
    print(len(dec_regs))