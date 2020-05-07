import os

import pandas as pd
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from pycocotools.coco import COCO

from dataset_utils import register_polyp_datasets, dataset_annots


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None, old_metric=False):
        self.iou_thresh = 0.0
        self.eval_mode = 'new'
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join("datasets", self.dataset_name)
        coco_annot_file = os.path.join(self.dataset_folder, "annotations", dataset_annots[dataset_name])
        self._coco_api = COCO(coco_annot_file)

        self.output_folder = os.path.join(output_dir, "giana")
        self.detection_folder = os.path.join(output_dir, "detection")
        self.localization_folder = os.path.join(output_dir, "localization")
        self.classification_folder = os.path.join(output_dir, "classification")
        self.old_metric = old_metric
        self.debug = False

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self._partial_results = []

        self.make_dirs()

        self.classes_id = MetadataCatalog.get(dataset_name).get("thing_dataset_id_to_contiguous_id")
        self.class_id_name = {v: k for k, v in
                              zip(MetadataCatalog.get(dataset_name).get("thing_classes"), self.classes_id.values())}

    def make_dirs(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)
        if not os.path.exists(self.localization_folder):
            os.makedirs(self.localization_folder)
        if not os.path.exists(self.classification_folder):
            os.makedirs(self.classification_folder)

    def reset(self):
        self.results = pd.DataFrame(columns=["image", "detected", "localized", "classified", "score", "pred_box"])
        self._partial_results = []

    def evaluate(self):

        if not self.debug:
            self.results = pd.DataFrame(self._partial_results,
                                        columns=["image", "detected", "localized", "classified", "score", "pred_box"])
        print(len(self._partial_results))
        print(self.results.groupby("image"))
        print(self.results.groupby("image").count())

        self.results[['sequence', 'frame']] = self.results.image.str.split("-", expand=True)
        sequences = pd.unique(self.results.sequence)
        dets = []
        locs = []
        classifs = []
        avg_df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        avg_df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        avg_df_classification = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        for sequence in sequences:
            df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
            df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
            df_classification = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
            filtered = self.results[self.results.sequence == sequence]
            filtered_det = self.results[self.results.sequence == sequence].drop_duplicates(subset="image")
            for threshold in self.thresholds:
                th_cond = (filtered.score >= threshold) | (filtered.score == -1)
                over_threshold = filtered[th_cond]
                under_threshold = filtered[~th_cond]

                over_threshold_det = filtered_det[th_cond].drop_duplicates(subset="image")
                under_threshold_det = filtered_det[~th_cond].drop_duplicates(subset="image")

                det = over_threshold_det.detected.value_counts()
                under_det = under_threshold_det.detected.value_counts()
                det_tp = det.TP if "TP" in det.keys() else 0
                det_fp = det.FP if "FP" in det.keys() else 0
                det_tn = (det.TN if "TN" in det.keys() else 0) + (under_det.FP if "FP" in under_det.keys() else 0)
                det_fn = (det.FN if "FN" in det.keys() else 0) + (under_det.TP if "TP" in under_det.keys() else 0)
                first_polyp = over_threshold_det[over_threshold_det.detected == "FN"].frame.apply(
                    lambda x: int(x.split(".")[0])).min()
                first_det_polyp = over_threshold_det[over_threshold_det.detected == "TP"].frame.apply(
                    lambda x: int(x.split(".")[0]))
                first_det_polyp = first_det_polyp[first_det_polyp >= first_polyp].min()
                det_rt = first_det_polyp - first_polyp
                self._add_row(df_detection, [threshold, det_tp, det_fp, det_tn, det_fn, det_rt])

                loc = over_threshold.localized.value_counts()
                under_loc = under_threshold.localized.value_counts()
                loc_tp = loc.TP if "TP" in loc.keys() else 0
                loc_fp = loc.FP if "FP" in loc.keys() else 0

                loc_tn = (loc.TN if "TN" in loc.keys() else 0) + (under_loc.FP if "FP" in under_loc.keys() else 0)
                loc_fn = (loc.FN if "FN" in loc.keys() else 0) + (under_loc.TP if "TP" in under_loc.keys() else 0)
                first_polyp = over_threshold[over_threshold.localized == "FN"].frame.apply(
                    lambda x: int(x.split(".")[0])).min()
                first_loc_polyp = over_threshold[over_threshold.localized == "TP"].frame.apply(
                    lambda x: int(x.split(".")[0]))
                first_loc_polyp = first_loc_polyp[first_loc_polyp >= first_polyp].min()
                loc_rt = first_loc_polyp - first_polyp
                self._add_row(df_localization, [threshold, loc_tp, loc_fp, loc_tn, loc_fn, loc_rt])

                clasif = over_threshold[over_threshold.localized == "TP"].classified.value_counts()

                class_tp = clasif.TP if "TP" in clasif.keys() else 0
                class_fp = clasif.FP if "FP" in clasif.keys() else 0
                class_tn = clasif.TN if "TN" in clasif.keys() else 0
                class_fn = clasif.FN if "FN" in clasif.keys() else 0
                self._add_row(df_classification, [threshold, class_tp, class_fp, class_tn, class_fn])

            df_detection.to_csv(
                os.path.join(self.detection_folder, "d{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                index=False)
            df_localization.to_csv(
                os.path.join(self.localization_folder, "l{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                index=False)
            df_classification.to_csv(os.path.join(self.classification_folder,
                                                  "c{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                                     index=False)
            dets.append(df_detection)
            locs.append(df_localization)
            classifs.append(df_classification)
        print("computing Averages and aggregation metrics")
        for det, loc, classif in zip(dets, locs, classifs):
            avg_df_detection = pd.concat([avg_df_detection, det], ignore_index=True, sort=False)
            avg_df_localization = pd.concat([avg_df_localization, loc], ignore_index=True, sort=False)
            avg_df_classification = pd.concat([avg_df_classification, classif], ignore_index=True, sort=False)

        self.compute_average_metrics(avg_df_detection, len(sequences), self.detection_folder)
        self.compute_average_metrics(avg_df_localization, len(sequences), self.localization_folder)
        self.compute_average_metrics(avg_df_classification, len(sequences), self.classification_folder)

        self.results.to_csv(os.path.join(self.output_folder, "results{}.csv".format("_old" if self.old_metric else "")),
                            index=False)

    def compute_average_metrics(self, df, sequences, save_folder):
        df = df.groupby("threshold")
        if "RT" in df.sum().columns:
            stdRT = df.std().RT
            df = df.sum()
            df['mRT'] = df.RT.apply(lambda x: round(x / sequences, 2))
            df['stdRT'] = stdRT.round(2)
        else:
            df = df.sum()
        df = self._compute_aggregation_metrics(df)
        df.to_csv(os.path.join(save_folder, "avg{}.csv".format("_old" if self.old_metric else "")), index=False)

    def _compute_aggregation_metrics(self, df):
        tp = df.TP
        fp = df.FP
        tn = df.TN
        fn = df.FN

        acc = (tp + tn) / (tp + fp + tn + fn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1core = 2 * pre * rec / (pre + rec)

        df['accuracy'] = acc.round(4)
        df["precision"] = pre.round(4)
        df["recall"] = rec.round(4)
        df["f1score"] = f1core.round(4)

        return df

    def _add_row(self, df, row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    def process(self, input, output):
        previous_len = len(self._partial_results)
        for instance, output in zip(input, output):
            input_image_id = instance['image_id']

            instance_gt_annots = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=input_image_id))

            im_name = os.path.basename(instance['file_name'])

            fields = output["instances"].get_fields()
            pred_boxes = fields['pred_boxes']  # xyxy
            scores = fields['scores'].cpu().numpy()
            pred_class = fields['pred_classes']

            if instance_gt_annots:
                # GT but not preds --> FN
                if len(pred_boxes) == 0:
                    for annot_dict in instance_gt_annots:
                        row = [im_name, "FN", "FN", "non-eval", -1, "NA"]
                        self._partial_results += [row]
                # GT and preds --> TP or FP
                else:
                    det_out = "TP"
                    from detectron2.structures import Boxes, pairwise_iou, BoxMode
                    gt_boxes = torch.tensor([annot_dict['bbox'] for annot_dict in instance_gt_annots])
                    gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    gt_boxes = Boxes(gt_boxes.to(pred_boxes.device))
                    ious = pairwise_iou(gt_boxes, pred_boxes)
                    paired_preds = []
                    for gt_idx, matches in enumerate(ious):
                        if matches.sum() == 0:
                            row = [im_name, "FN", "FN", "non-eval", -1, "NA"]
                            self._partial_results += [row]
                        else:
                            if self.eval_mode == "iou":
                                pred_idx = matches.argmax()
                                if pred_idx not in paired_preds:
                                    paired_preds.append(pred_idx)
                                    class_out = self._is_polyp_classified(pred_class[pred_idx], instance_gt_annots[gt_idx]['category_id'])
                                    row = [im_name, det_out, "TP", class_out, scores[pred_idx], pred_boxes[pred_idx]]
                                    self._partial_results += [row]
                                else:
                                    row = [im_name, det_out, "FP", "non-eval", scores[pred_idx], pred_boxes[pred_idx]]
                                    self._partial_results += [row]
                            else:
                                for posible_match in matches.nonzero():
                                    gt_box = gt_boxes.tensor[gt_idx]
                                    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                                    pred_box = pred_boxes.tensor[posible_match]
                                    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box.squeeze()

                                    if self.eval_mode == 'old':
                                        pred_cx, pred_cy = (pred_x1 + (pred_x2 - pred_x1) / 2), (pred_y1 + (pred_y2 - pred_y1) / 2)
                                        eval_condition = (gt_x1 < pred_cx < gt_x2) and (gt_y1 < pred_cy < gt_y2)
                                    else:
                                        gt_cx, gt_cy = (gt_x1 + (gt_x2 - gt_x1) / 2), (gt_y1 + (gt_y2 - gt_y1) / 2)
                                        eval_condition = (pred_x1 < gt_cx < pred_x2) and (pred_y1 < gt_cy < pred_y2)

                                    if eval_condition:
                                        if posible_match not in paired_preds:
                                            paired_preds.append(posible_match)
                                            class_out = self._is_polyp_classified(pred_class[posible_match],instance_gt_annots[gt_idx]['category_id'])
                                            row = [im_name, det_out, "TP", class_out, scores[posible_match], pred_boxes[posible_match]]
                                            self._partial_results += [row]
                                        else:
                                            row = [im_name, det_out, "FP", "non-eval", scores[posible_match], pred_boxes[posible_match]]
                                            self._partial_results += [row]



                    # for pred_box, pred_score, pred_classif in zip(pred_boxes, scores, pred_class):
                    #     pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
                    #     if instance_gt_annots:
                    #         for annot_dict in instance_gt_annots:
                    #             gt_bbox = annot_dict['bbox']  # xywh
                    #             gt_bbox[2] += gt_bbox[0]
                    #             gt_bbox[3] += gt_bbox[1]  # xyxy
                    #
                    #             gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
                    #
                    #             eval_condition = self._is_localized(gt_bbox, gt_x1, gt_x2, gt_y1, gt_y2, pred_box,
                    #                                                 pred_x1, pred_x2, pred_y1, pred_y2)
                    #
                    #             if eval_condition:
                    #                 class_out = self._is_polyp_classified(pred_classif, annot_dict['category_id'])
                    #
                    #                 row = [im_name, det_out, "TP", class_out, pred_score, pred_box]
                    #                 self._partial_results += [row]
                    #                 instance_gt_annots.remove(annot_dict)
                    #                 break
                    #
                    #     else:
                    #         row = [im_name, "FP", "FP", "non-eval", pred_score, pred_box]
                    #         self._partial_results += [row]
            else:
                # No GT but Preds --> FP
                if len(pred_boxes) > 0:
                    for pred_box, pred_score, pred_classif in zip(pred_boxes, scores, pred_class):
                        row = [im_name, "FP", "FP", "non-eval", pred_score, pred_box]
                        self._partial_results += [row]
                # No GT and no Preds --> TN
                else:
                    row = [im_name, "TN", "TN", "non-eval", -1, "NA"]
                    self._partial_results += [row]

    def _is_localized(self, gt_bbox, gt_x1, gt_x2, gt_y1, gt_y2, pred_box, pred_x1, pred_x2, pred_y1, pred_y2):
        if self.eval_mode == 'iou':
            eval_condition = bb_intersection_over_union(gt_bbox,
                                                        pred_box.tensor.numpy()) > self.iou_thresh
        elif self.eval_mode == 'old':
            pred_cx, pred_cy = (pred_x1 + (pred_x2 - pred_x1) / 2), (
                    pred_y1 + (pred_y2 - pred_y1) / 2)
            eval_condition = (gt_x1 < pred_cx < gt_x2) and (gt_y1 < pred_cy < gt_y2)
        else:
            gt_cx, gt_cy = (gt_x1 + (gt_x2 - gt_x1) / 2), (gt_y1 + (gt_y2 - gt_y1) / 2)
            eval_condition = (pred_x1 < gt_cx < pred_x2) and (pred_y1 < gt_cy < pred_y2)
        return eval_condition

    @staticmethod
    def _is_polyp_classified(pred, gt):
        if pred + gt == 2:
            return "TP"
        if pred + gt == 0:
            return "TN"
        if pred == 1:
            return "FP"
        else:
            return "FN"


def offline_evaluation(dataset_name, output_dir, results_file):
    evaluator = GianaEvaulator(dataset_name, output_dir)
    evaluator.results = pd.read_csv(results_file)
    evaluator.debug = True
    evaluator.evaluate()

def compute_from_coco_results(coco_gt, coco_preds, eval_mode='new'):
    partial_results = []
    for image_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(image_id)
        im_name = img_info['file_name']
        gt_anns_ids = coco_gt.getAnnIds(imgIds=image_id)
        pred_anns_ids = coco_preds.getAnnIds(imgIds=image_id)

        gt_anns = coco_gt.loadAnns(gt_anns_ids)
        pred_anns = coco_preds.loadAnns(pred_anns_ids)


        if gt_anns:
            if len(pred_anns) == 0:
                for annot_dict in gt_anns:
                    row = [im_name, "FN", "FN", "non-eval", -1, "NA"]
                    partial_results += [row]

            else:
                det_out = "TP"

                for prediction in pred_anns:
                    pred_x1, pred_y1, pred_x2, pred_y2 = prediction['bbox']
                    pred_x2 += pred_x1
                    pred_y2 += pred_y1  # xyxy
                    if gt_anns:
                        for annot_dict in gt_anns:
                            gt_bbox = annot_dict['bbox']  # xywh
                            gt_bbox[2] += gt_bbox[0]
                            gt_bbox[3] += gt_bbox[1]  # xyxy

                            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

                            if eval_mode == 'iou':
                                eval_condition = bb_intersection_over_union(gt_bbox,
                                                                            pred_box.tensor.numpy()) > self.iou_thresh
                            elif eval_mode == 'old':
                                pred_cx, pred_cy = (pred_x1 + (pred_x2 - pred_x1) / 2), (
                                        pred_y1 + (pred_y2 - pred_y1) / 2)
                                eval_condition = (gt_x1 < pred_cx < gt_x2) and (gt_y1 < pred_cy < gt_y2)
                            else:
                                gt_cx, gt_cy = (gt_x1 + (gt_x2 - gt_x1) / 2), (gt_y1 + (gt_y2 - gt_y1) / 2)
                                eval_condition = (pred_x1 < gt_cx < pred_x2) and (pred_y1 < gt_cy < pred_y2)

                            if eval_condition:
                                class_out = self._is_polyp_classified(pred_classif, annot_dict['category_id'])

                                row = [im_name, det_out, "TP", class_out, pred_score, pred_box]
                                partial_results += [row]
                                instance_gt_annots.remove(annot_dict)
                                break
                    else:
                        row = [im_name, "FP", "FP", "non-eval", pred_score, pred_box]
                        partial_results += [row]

        else:
            if len(pred_anns) > 0:
                for pred_box, pred_score, pred_classif in zip(pred_boxes, scores, pred_class):
                    row = [im_name, "FP", "FP", "non-eval", pred_score, pred_box]
                    partial_results += [row]

            else:
                row = [im_name, "TN", "TN", "non-eval", -1, "NA"]
                partial_results += [row]

    results = pd.DataFrame(partial_results, columns=["image", "detected", "localized", "classified", "score", "pred_box"])


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = (interArea + 1e-5) / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def eval_giana(coco_gt, coco_preds, eval_condition_mode="new", iou_threshold=0.0):
    tp, fp, tn, fn = 0, 0, 0, 0
    for image_id in coco_gt.getImgIds():
        gt_anns_ids = coco_gt.getAnnIds(imgIds=image_id)
        pred_anns_ids = coco_preds.getAnnIds(imgIds=image_id)

        gt_anns = coco_gt.loadAnns(gt_anns_ids)
        pred_anns = coco_preds.loadAnns(pred_anns_ids)
        if gt_anns:
            if pred_anns:
                diff_boxes = len(gt_anns) - len(pred_anns)
                if diff_boxes < 0:
                    fp += -diff_boxes
                elif diff_boxes > 0:
                    fn += diff_boxes

                for gt_annot in gt_anns:
                    gt_box = gt_annot['bbox']
                    for pred_annot in pred_anns:
                        pred_box = pred_annot['bbox']

                        if eval_condition_mode == 'iou':
                            eval_condition = bb_intersection_over_union(gt_box, pred_box) > iou_threshold
                        elif eval_condition_mode == 'old':
                            pred_cx, pred_cy = (pred_box[0] + (pred_box[2] - pred_box[0]) / 2), (
                                    pred_box[1] + (pred_box[3] - pred_box[1]) / 2)
                            eval_condition = (gt_box[0] < pred_cx < gt_box[2]) and (gt_box[1] < pred_cy < gt_box[3])
                        elif eval_condition_mode == 'new':
                            gt_cx, gt_cy = (gt_box[0] + (gt_box[2] - gt_box[0]) / 2), (
                                    gt_box[1] + (gt_box[3] - gt_box[1]) / 2)
                            eval_condition = (pred_box[0] < gt_cx < pred_box[2]) and (pred_box[1] < gt_cy < pred_box[3])

                        if eval_condition:
                            tp += 1
                            pred_anns.pop()
                        else:
                            fp += 1
            else:
                fn += len(gt_anns)
        else:
            if pred_anns:
                fp += len(pred_anns)
            else:
                tn += 1

    print(tp, fp, tn, fn)


if __name__ == '__main__':
    from argparse import ArgumentParser

    register_polyp_datasets()
    ap = ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--file")
    opts = ap.parse_args()

    offline_evaluation("CVC_VideoClinicDB_test", "/home/devsodin/PycharmProjects/centermask2/output/CenterMask-R-50-FPN-ms-3x/inference", "/home/devsodin/PycharmProjects/centermask2/output/CenterMask-R-50-FPN-ms-3x/inference/giana/results.csv")

    coco_gt = COCO("datasets/CVC_VideoClinicDB_valid/annotations/valid.json")
    coco_gt = COCO("datasets/CVC_VideoClinicDB_test/annotations/test.json")
    # coco_res = coco_gt.loadRes()

    det = "/home/devsodin/PycharmProjects/detectron2/results/trident_da_adj/inference/coco_instances_results.json"
    # det = "output/CenterMask-R-50-FPN-ms-3x/inference/val.json"
    coco_preds = coco_gt.loadRes(det)

    print('new')
    eval_giana(coco_gt, coco_preds, 'new')
    print('old')
    eval_giana(coco_gt, coco_preds, 'old')
    print('iou')
    eval_giana(coco_gt, coco_preds, 'iou', iou_threshold=0.)
