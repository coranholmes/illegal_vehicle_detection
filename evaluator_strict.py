import os, argparse, json
import numpy as np
from utils import *


def calculate_f1(tp, fp, fn):
    text = ""
    if tp + fp != 0 and tp + fn != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        text = text + "tp: " + str(tp) + "\tfp: " + str(fp) + "\tfn: " + str(fn) + "\n"
        text = (
            text
            + "precision: "
            + str(precision)
            + "\trecall: "
            + str(recall)
            + "\tf1: "
            + str(f1)
            + "\n"
        )
    else:
        text = text + "tp: " + str(tp) + "\tfp: " + str(fp) + "\tfn: " + str(fn) + "\n"
        text += "No precision, recall, f1 calculated!\n"
    return text


def evaluate_exp(args):
    label_path, ds_name, mode = args.path, args.name, args.mode
    if mode == "frame":
        by_frame = True
    else:
        by_frame = False

    if os.path.isfile(label_path):  # process single video
        print("Processing single label_path: %s" % label_path)
        label_dir, label_name = os.path.split(label_path)
        ds_root = os.path.abspath(os.path.join(label_dir, ".."))
        gt_path = os.path.join(ds_root, "gt", label_name)
        tp, fp, fn = process_label(label_path, gt_path, by_frame)
        calculate_f1(tp, fp, fn)
    else:
        ds_root = os.path.join(os.getcwd(), "videos", ds_name)
        label_dir = os.path.join(ds_root, "label")
        exp_path = os.path.join(ds_root, "exp", "eval_res.txt")
        exp_file = open(exp_path, "a")
        tp, fp, fn = 0, 0, 0
        for file_name in os.listdir(label_dir):
            label_path = os.path.join(label_dir, file_name)
            print("Evaluating on " + label_path)
            gt_path = os.path.join(ds_root, "gt", file_name)
            tp1, fp1, fn1 = process_label(label_path, gt_path, by_frame)
            tp += tp1
            fp += fp1
            fn += fn1
        text = calculate_f1(tp, fp, fn)
        exp_file.writelines(ds_name + "_" + mode + "_" + get_exp_paras() + "\n")
        exp_file.writelines(text + "\n")
        print(text)


def process_label(label_path, gt_path, by_frame=False):
    print("Processing label file: %s" % label_path)
    label_file = open(label_path)
    gt_file = open(gt_path)

    gts = dict()
    gt_ids_set = set()
    for line in gt_file:
        # {"frame": 210, "id": 6, "type": "bus", "top": 212, "left": 117, "bottom": 343, "right": 286}
        d = json.loads(line)
        if d["frame"] not in gts:
            gts[d["frame"]] = dict()
        gts[d["frame"]][d["id"]] = [d["top"], d["left"], d["bottom"], d["right"]]
        gt_ids_set.add(d["id"])

    decs = dict()
    p = 0
    dec_ids_set = set()
    for line in label_file:
        # {"frame": 210, "id": 6, "type": "bus", "top": 212, "left": 117, "bottom": 343, "right": 286, "parked_time": 5, "detected": "YES"}
        d = json.loads(line)
        if d["detected"] == "YES":  # TODO positives做预处理删除特别小的框
            p += 1
            dec_ids_set.add(d["id"])
            if d["frame"] not in decs:
                decs[d["frame"]] = dict()
            decs[d["frame"]][d["id"]] = [
                d["top"],
                d["left"],
                d["bottom"],
                d["right"],
            ]
    p2 = len(
        dec_ids_set
    )  # p is the number of detected bounding boxes, p2 is the number of detected ids
    print("detected ids:", dec_ids_set, "\tgt ids:", gt_ids_set)

    tp, fn = 0, 0
    if (
        by_frame
    ):  # evaluate based on frames, every gt box in one frame is regarded as one positive sample
        for frame in sorted(gts):  # 遍历gt中的每个frame
            cur_frame_gt_cnt = len(gts[frame])
            match_cnt = 0
            for gt_id in gts[frame]:  #  遍历gt中的每个frame中的每个box
                if frame in decs:
                    for dec_id in decs[
                        frame
                    ]:  # 对于对应frame中每个detection到的box进行匹配(det 匹配 gt)
                        iou = get_iou(gts[frame][gt_id], decs[frame][dec_id])
                        if iou > EVALUATION_IOU_THRESHOLD:
                            tp += 1
                            match_cnt += 1
                            break  # gt匹配到一个dec就不再继续，跳出循环
            if frame not in decs:  # fn
                print(frame, gts[frame].keys(), [])
            elif cur_frame_gt_cnt != len(decs[frame]):  # fn + 一部分fp
                print(frame, gts[frame].keys(), decs[frame].keys())
            fn = fn + (cur_frame_gt_cnt - match_cnt)  # fn就是没匹配到的
        fp = p - tp
    else:  # evaluate based on events
        gt_ids = []
        dec_ids = []
        for frame in sorted(gts):  # 遍历gt中的每个frame
            cur_frame_gt_cnt = len(gts[frame])
            match_cnt = 0
            for gt_id in gts[frame]:  #  遍历gt中的每个frame中的每个box
                if gt_id in gt_ids:  # 如果gt已经匹配过就跳过
                    continue
                if frame in decs:
                    for dec_id in decs[frame]:
                        iou = get_iou(gts[frame][gt_id], decs[frame][dec_id])
                        if iou > EVALUATION_IOU_THRESHOLD:
                            tp += 1
                            match_cnt += 1
                            gt_ids.append(gt_id)
                            dec_ids.append(dec_id)
                            print(dec_id, "matches", gt_id)
        if p2 < len(dec_ids):
            print("============================", p2, len(dec_ids))
            p2 = len(dec_ids)
        fp = p2 - tp
        fn = len(gt_ids_set) - tp

    print("p:", p if by_frame else p2, "tp:", tp, "fp:", fp, "fn:", fn)
    return tp, fp, fn


def get_iou(gt, dec):
    gt = np.array(gt)
    dec = np.array(dec)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], dec[0])
    yA = max(gt[1], dec[1])
    xB = min(gt[2], dec[2])
    yB = min(gt[3], dec[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (dec[2] - dec[0] + 1) * (dec[3] - dec[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate results!")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="xd_full",
        help="ISLab|xd_full, choose the dataset to evaluate the experiment",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="videos/xd_full/label/sunny2.txtt",
        help="choose a label to process.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="frame",
        help="frame|event, choose the evaluation mode",
    )
    # parser.add_argument('-p', "--path", type=str, default="videos/ISLab/input/ISLab-04.mp4", help="choose a video to be processed")
    args = parser.parse_args()
    evaluate_exp(args)
