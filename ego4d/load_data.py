import json

STA_TRAIN_PATH = "/workspaces/content/ego4d_data/v1/annotations/fho_sta_train.json"
STA_VAL_PATH = "/workspaces/content/ego4d_data/v1/annotations/fho_sta_val.json"
STA_TEST_PATH = "/workspaces/content/ego4d_data/v1/annotations/fho_sta_test_unannotated.json"
OBJ_DET_PATH = "/workspaces/content/ego4d_data/v1/sta_models/object_detections.json"

metadata_file_path = "/workspaces/content/ego4d_data/ego4d.json"
ego4d_meta = json.load(open(metadata_file_path))

video_to_dims = {
    v["video_uid"]: {
        "frame_height": v["video_metadata"]["display_resolution_height"],
        "frame_width": v["video_metadata"]["display_resolution_width"],
    }
    for v in ego4d_meta["videos"]
}
len(video_to_dims)

sta_train = json.load(open(STA_TRAIN_PATH))
train_video_uids = set(x.get("video_id", x.get("video_uid", None)) for x in sta_train["annotations"])

sta_val = json.load(open(STA_VAL_PATH))
val_video_uids = set(x.get("video_id", x.get("video_uid", None)) for x in sta_val["annotations"])

sta_test = json.load(open(STA_TEST_PATH))
test_video_uids = set(x.get("video_id", x.get("video_uid", None)) for x in sta_test["annotations"])

# Features Preprocessing

import os
from typing import Any, Callable, List, Optional, Tuple

import h5py
import torch
from tqdm.auto import tqdm

def save_ego4d_features_to_hdf5(video_uids: List[str], feature_dir: str, out_path: str):
    """
    Use this function to preprocess Ego4D features into a HDF5 file with h5py
    """
    errors = []
    with h5py.File(out_path, "w") as out_f:
        for uid in tqdm(video_uids, desc="video_uid", leave=True):
            feature_path = os.path.join(feature_dir, f"{uid}.pt")
            if not os.path.exists(feature_path):
                errors.append(uid)
                continue
            fv = torch.load(feature_path)
            out_f.create_dataset(uid, data=fv.numpy())
    return errors

train_feature_path = "/workspaces/content/features_train.hdf5"
train_err_video_uids = save_ego4d_features_to_hdf5(train_video_uids, feature_dir="/workspaces/content/ego4d_data/v1/omnivore_video_swinl_fp16", out_path=train_feature_path)

val_feature_path = "/workspaces/content/features_val.hdf5"
val_err_video_uids = save_ego4d_features_to_hdf5(val_video_uids, feature_dir="/workspaces/content/ego4d_data/v1/omnivore_video_swinl_fp16", out_path=val_feature_path)

test_feature_path = "/workspaces/content/features_test.hdf5"
test_err_video_uids = save_ego4d_features_to_hdf5(test_video_uids, feature_dir="/workspaces/content/ego4d_data/v1/omnivore_video_swinl_fp16", out_path=test_feature_path)

train_video_uids_to_use = list(set(train_video_uids) - set(train_err_video_uids))
val_video_uids_to_use = list(set(val_video_uids) - set(val_err_video_uids))
test_video_uids_to_use = list(set(test_video_uids) - set(test_err_video_uids))

print(len(train_video_uids_to_use), len(set(train_err_video_uids)))

# Declare PyTorch Dataset

import torch
from torch.utils.data import Dataset
import numpy as np
import json

import functools


def _one_hot_encoding(n, clazzes):
    result = torch.zeros(n)
    result[clazzes] = 1
    return result


class STAFeatureDataset(Dataset):
    """
    This dataset loads STA data via loading the corresponding:
    - Feature vector for the clip
    - Pre-detected bounding boxes
    - Ground truth bounding boxes, verbs, nouns and time to contact (ttc)
    """
    def __init__(self, video_uids, data, feature_path, obj_det_path, det_score_threshold, next_active_threshold, keep_max_iou):
        self.anns = [x for x in data.get("annotations", data.get("clips", None)) if x.get("video_id", x.get("video_uid", None)) in video_uids]
        self.features = h5py.File(feature_path)
        self.num_nouns = len(data["noun_categories"])
        self.num_verbs = len(sta_train["verb_categories"])
        self.obj_dets = json.load(open(obj_det_path))
        self.det_score_threshold = det_score_threshold
        self.next_active_threshold = next_active_threshold
        self.keep_max_iou = keep_max_iou
    
    def __len__(self):
        return len(self.anns)
    
    def _get_obj_dets(self, uid, w, h):
        object_detections = self.obj_dets[uid]

        # taken from:
        # https://github.com/EGO4D/forecasting/blob/main/ego4d/datasets/short_term_anticipation.py#L699
        if len(object_detections) > 0:
            pred_boxes = np.vstack([[
                x['box'][0] / w,
                x['box'][1] / h,
                x['box'][2] / w,
                x['box'][3] / h
              ] for x in object_detections]
            )
            pred_scores = np.array([x['score'] for x in object_detections])
            pred_object_labels = np.array([x['noun_category_id'] for x in object_detections])

            # exclude detections below the theshold
            detected = (
                pred_scores
                >= self.det_score_threshold
            )

            pred_boxes = pred_boxes[detected]
            pred_object_labels = pred_object_labels[detected]
            pred_scores = pred_scores[detected]
        else:
            pred_boxes = np.zeros((0, 4))
            pred_scores = pred_object_labels = np.array([])

        return {
            "pred_boxes": torch.tensor(pred_boxes).to(torch.float32), 
            "pred_object_labels": torch.tensor(pred_object_labels).to(torch.float32), 
            "pred_scores": torch.tensor(pred_scores).to(torch.float32),
        }
    
    def __getitem__(self, idx):
        ann = self.anns[idx]
        uid = ann["uid"]
        v_uid = ann.get("video_id", ann.get("video_uid", None))
        start_idx = ann["frame"] // 16   # TODO use start time
        fs = torch.tensor(self.features[v_uid][start_idx]).to(torch.float32)
        dims = video_to_dims[v_uid]
        w, h = dims["frame_width"], dims["frame_height"]

        boxes = []
        verbs = []
        nouns = []
        contact_time = []
        num_objs = len(ann.get("objects", []))
        for obj in ann.get("objects", []):
            boxes.append(torch.tensor([
                float(obj["box"][0] / w),
                float(obj["box"][1] / h), 
                float(obj["box"][2] / w),
                float(obj["box"][3] / h),
            ]))
            verbs.append(_one_hot_encoding(self.num_verbs, obj["verb_category_id"]))
            nouns.append(_one_hot_encoding(self.num_nouns, obj["noun_category_id"]))
            contact_time.append(torch.tensor(float(obj["time_to_contact"])))
        
        label_dict = {
            "boxes": torch.stack(boxes) if len(boxes) > 0 else torch.empty(1),
            "verbs": torch.stack(verbs) if len(verbs) > 0 else torch.empty(1),
            "nouns": torch.stack(nouns) if len(nouns) > 0 else torch.empty(1),
            "ttc": torch.stack(contact_time) if len(contact_time) > 0 else torch.empty(1),
            "num_objs": torch.tensor(num_objs),
            "uids": ann["uid"],
            "video_uids": ann.get("video_id", ann.get("video_uid", None)),
        }

        pred = self._get_obj_dets(uid, w, h)

        # labelled
        if len(boxes) > 0:
            ious = compute_iou(pred["pred_boxes"], label_dict["boxes"])
            matches = ious.argmax(-1)
            ious = ious.max(-1)
            if self.keep_max_iou:
                next_active = ious >= min(ious.max() - 1e-2, self.next_active_threshold)
            else:
                next_active = ious >= self.next_active_threshold

            # filter out next active instead of setting to nan
            pred["pred_verbs"] = label_dict["verbs"][matches]
            pred["pred_nouns"] = label_dict["nouns"][matches]
            pred["pred_ttc"] = label_dict["ttc"][matches]
            for k in "pred_verbs", "pred_ttc", "pred_boxes", "pred_scores", "pred_object_labels", "pred_nouns":
                pred[k] = pred[k][next_active]

        label_dict.update(pred)
        return fs, label_dict

# Declare Model

import torch.nn as nn

class StaFeaturesModel(nn.Module):
    def __init__(self, in_feature_dim, proj_feature_dim, num_nouns, num_verbs, leaky=0.2):
        super().__init__()
        # TODO: self attention / non local block
        self.proj = nn.Sequential(
            nn.Linear(in_feature_dim, proj_feature_dim),
            nn.ReLU(True),
            # nn.Linear(proj_feature_dim, proj_feature_dim),
            # nn.LeakyReLU(leaky),
        )
        self.roi_head = nn.Sequential(
            nn.Linear(proj_feature_dim + 4, proj_feature_dim),
            # nn.LeakyReLU(leaky),
            nn.ReLU(True),
        )
        self.verb_head = nn.Linear(proj_feature_dim, num_verbs)
        self.noun_head = nn.Linear(proj_feature_dim, num_nouns)
        self.ttc_head = nn.Linear(proj_feature_dim, 1)
        self.apply(self.init_weights)

    def forward(self, x, boxes):
        bs = x.shape[0]
        if boxes.shape[0] == 0:
            return {
                "nouns": torch.zeros((0, self.noun_head.out_features)),
                "verbs": torch.zeros((0, self.verb_head.out_features)),
                "ttc": torch.zeros((0, self.ttc_head.out_features)),
            }
            
        assert len(boxes.shape) == 2 and boxes.shape[1] == 5
        box_idx = boxes[:, 0].long()
        box_vals = boxes[:, 1:]

        p = self.proj(x)
        p = p[box_idx]  # dupe rows for examples with multiple boxes
        pb = torch.cat((p, box_vals), dim=-1)

        r = self.roi_head(pb)
        n = self.noun_head(r)
        v = self.verb_head(r)
        ttc = self.ttc_head(r)

        if not self.training:
            n = F.softmax(n, dim=-1)
            v = F.softmax(v, dim=-1)
        return {"nouns": n, "verbs": v, "ttc": ttc.squeeze()}
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(
                module.weight.data, gain=torch.nn.init.calculate_gain("relu")
            )
            module.bias.data.zero_()

# Eval Utils

# This is copied from https://github.com/EGO4D/forecasting/blob/main/ego4d/evaluation/sta_metrics.py
# Why? Because of a package name conflict - both repositories have `ego4d` as the prefix. This needs to change.

import numpy as np
from abc import ABC, abstractmethod


def compute_iou(preds,gts):
    """
    Compute a matrix of intersection over union values for two lists of bounding boxes using broadcasting
    :param preds: matrix of predicted bounding boxes [NP x 4]
    :param gts: number of ground truth bounding boxes [NG x 4]
    :return: an [NP x NG] matrix of IOU values
    """
    # Convert shapes to use broadcasting
    # preds: NP x 4 -> NP x 1 x 4
    # gts: NG x 4 -> 1 x NG x 4
    preds = np.expand_dims(preds,1)
    gts = np.expand_dims(gts,0)

    def area(boxes):
        width = boxes[..., 2] - boxes[..., 0] + 1
        height = boxes[..., 3] - boxes[..., 1] + 1
        width[width<0]=0
        height[height<0]=0
        return width * height

    ixmin = np.maximum(gts[..., 0], preds[..., 0])
    iymin = np.maximum(gts[..., 1], preds[..., 1])
    ixmax = np.minimum(gts[..., 2], preds[..., 2])
    iymax = np.minimum(gts[..., 3], preds[..., 3])

    areas_preds = area(preds)
    areas_gts = area(gts)
    areas_intersections = area(np.stack([ixmin, iymin, ixmax, iymax], -1))

    return (areas_intersections) / (areas_preds + areas_gts - areas_intersections+1e-11)


class AbstractMeanAveragePrecision(ABC):
    """
    Abstract class for implementing mAP measures
    """
    def __init__(self, num_aps, percentages=True, count_all_classes=True, top_k=None):
        """
        Contruct the Mean Average Precision metric
        :param num_aps: number of average precision metrics to compute. E.g., we can compute different APs for different
                        IOU overlap thresholds
        :param percentages: whether the metric should return percentages (i.e., 0-100 range rather than 0-1)
        :param count_all_classes: whether to count all classes when computing mAP. If false, classes which do not have
                                    any ground truth label but do have associated predictions are counted (they will have
                                    an AP equal to zero), otherwise, only classes for which there is at least one ground truth
                                    label will count. It is useful to set this to True for imbalanced datasets for which not
                                    all classes are in the ground truth labels.
        :param top_k: the K to be considered in the top-k criterion. If None, a standard mAP will be computed
        """
        self.true_positives = []
        self.confidence_scores = []
        self.predicted_classes = []
        self.gt_classes = []

        self.num_aps = num_aps
        self.percentages = percentages
        self.count_all_classes = count_all_classes
        self.K = top_k
        self.names = []
        self.short_names = []

    def get_names(self):
        return self.names

    def get_short_names(self):
        return self.short_names

    def add(self,
            preds,
            labels
            ):
        """
        Add predictions and labels of a single image and matches predictions to ground truth boxes
        :param predictions: dictionary of predictions following the format below. While "boxes" and "scores" are
                            mandatory, other properties can be added (they can be used to compute matchings).
                            It can also be a list of dictionaries if predictions of more than one images are being added.
                {
                    'boxes' : [
                        [245,128,589,683],
                        [425,68,592,128]
                    ],
                    'scores' : [
                        0.8,
                        0.4
                    ],
                    'nouns' : [
                        3,
                        5
                    ],
                    'verbs': [
                        8,
                        11
                    ],
                    'ttcs': [
                        1.25,
                        1.8
                    ]
                }
        :param labels: dictionary of labels following a similar format. It can be a list of dictionaries.
                {
                    'boxes' : [
                        [195,322,625,800],
                        [150,300,425,689]
                    ],
                    'nouns' : [
                        9,
                        5
                    ],
                    'verbs': [
                        3,
                        11
                    ],
                    'ttcs': [
                        0.25,
                        1.25
                    ]
                }
        :return matched: a list of pairs of predicted/matched gt boxes
        """
        matched = []

        if len(preds) > 0:
            predicted_boxes = preds['boxes']
            predicted_scores = preds['scores']
            predicted_classes = self._map_classes(preds)

            # Keep track of correctly matched boxes for the different AP metrics
            true_positives = np.zeros((len(predicted_boxes), self.num_aps))

            if len(labels) > 0:
                # get GT boxes
                gt_boxes = labels['boxes']

                # IOU between all predictions and gt boxes
                ious = compute_iou(predicted_boxes, gt_boxes)

                # keep track of GT boxes which have already been matched
                gt_matched = np.zeros((len(gt_boxes), self.num_aps))

                # from highest to lowest score
                for i in predicted_scores.argsort()[::-1]:
                    # get overlaps related to this prediction
                    overlaps = ious[i].reshape(-1, 1)  # NGT x 1

                    # check if this prediction can be matched to the GT labels
                    # this will give different set of matchings for the different AP metrics
                    matchings = self._match({k: p[i] for k, p in preds.items()}, labels, overlaps)  # NGT x NR

                    # replicate overlaps to match shape of matching (different AP metrics)
                    overlaps = np.tile(overlaps, [1, matchings.shape[1]])  # NGT x NR

                    # do not allow to match a matched GT boxes
                    try:
                        matchings[gt_matched == 1] = 0  # not a valid match #NGT x NR
                    except:
                        import traceback; traceback.print_exc()
                        breakpoint()

                    # remove overlaps corresponding to boxes which are not a match
                    overlaps[matchings == 0] = -1

                    jj = overlaps.argmax(0)  # get indexes of maxima wrt GT

                    # get values of matching obtained at maxima
                    # these indicate if the matchings are correct
                    i_matchings = matchings[jj, range(len(jj))]

                    jj_matched = jj.copy()
                    jj_matched[~i_matchings] = -1

                    # set true positive to 1 if we obtained a matching
                    true_positives[i, i_matchings] = 1

                    # set the ground truth as matched if we obtained a matching
                    gt_matched[jj, range(len(jj))] += i_matchings

                    matched.append(jj_matched)

                # remove the K highest score false positives
                if self.K is not None and self.K>1:
                    # number of FP to remove:
                    K = (self.K - 1) * len(labels['boxes'])
                    # indexes to sort the predictions
                    order = predicted_scores.argsort()[::-1]
                    # sort the true positives labels
                    sorted_tp = (true_positives[order, :]).astype(float)
                    # invert to obtain the sorted false positive labels
                    sorted_fp = 1 - sorted_tp
                    # flag the first K false positives
                    sorted_tp[(sorted_fp.cumsum(0) <= K) & (sorted_fp == 1)] = np.nan

                    true_positives = sorted_tp
                    predicted_scores = predicted_scores[order]
                    predicted_classes = predicted_classes[order]

                self.gt_classes.append(self._map_classes(labels))

            # append list of true positives and confidence scores
            self.true_positives.append(true_positives)
            self.confidence_scores.append(predicted_scores)
            self.predicted_classes.append(predicted_classes)
        else:
            if len(preds) > 0:
                self.gt_classes.append(self._map_classes(labels))
        if len(matched) > 0:
            return np.stack(matched, 0)
        else:
            return np.zeros((0, self.num_aps))

    def _map_classes(self, preds):
        """
        Return the classes related to the predictions. These are used to specify how to compute mAP.
        :param preds: the predictions
        :return: num_ap x len(pred) array specifying the class of each prediction according to the different AP measures
        """
        return np.vstack([preds['nouns']] * self.num_aps).T

    def _compute_prec_rec(self, true_positives, confidence_scores, num_gt):
        """
        Compute precision and recall curve from a true positive list and the related scores
        :param true_positives: set of true positives
        :param confidence_scores:  scores associated to the true positives
        :param num_gt: number of ground truth labels for current class
        :return: prec, rec: lists of precisions and recalls
        """
        # sort true positives by confidence score
        tps = true_positives[confidence_scores.argsort()[::-1]]

        tp = tps.cumsum()
        fp = (1 - tps).cumsum()

        # safe division which turns x/0 to zero
        prec = self._safe_division(tp, tp + fp)
        rec = self._safe_division(tp, num_gt)

        return prec, rec

    def _safe_division(self, a, b):
        """
        Divide a by b avoiding a DivideByZero exception
        Inputs:
            a, b: either vectors or scalars
        Outputs:
            either a vector or a scalar
        """
        a_array = isinstance(a, np.ndarray)
        b_array = isinstance(b, np.ndarray)

        if (not a_array) and (not b_array):
            # both scalars
            # anything divided by zero should be zero
            if b == 0:
                return 0

        # numerator scalar, denominator vector
        if b_array and not a_array:
            # turn a into a vector
            a = np.array([a] * len(b))

        # numerator vector, denominator scalar
        if not b_array and a_array:
            # turn a into a vector
            b = np.array([b] * len(a))

        # turn all cases in which b=0 in a 0/1 division (result is 0)
        zeroden = b == 0
        b[zeroden] = 1
        a[zeroden] = 0
        return a / b

    def _compute_ap(self, prec, rec):
        """
        Python implementation of Matlab VOC AP code.
            1) Make precision monotonically decreasing 2) tThen compute AP by numerical integration.
        :param prec: vector of precision values
        :param rec: vector of recall values
        :return: average precision
        """
        # pad precision and recall
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], prec, [0]))

        # make precision monotonically decresing
        for i in range(len(mpre) - 2, 0, -1):
            mpre[i] = np.max((mpre[i], mpre[i + 1]))

        # consider only indexes in which the recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1

        # compute the area uner the curve
        return np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    def _compute_mr(self, prec, rec):
        """
        Compute maximum recall
        """
        return np.max(rec)

    def evaluate(self, measure='AP'):
        """
        Compute AP/MR for all classes, then averages
        """

        metrics = []
        # compute the different AP values for the different metrics

        gt_classes = np.concatenate(self.gt_classes)
        predicted_classes = np.concatenate(self.predicted_classes)
        true_positives = np.concatenate(self.true_positives)
        confidence_scores = np.concatenate(self.confidence_scores)

        for i in range(self.num_aps):
            # the different per-class AP values
            measures = []

            _gt_classes = gt_classes[:, i]
            _predicted_classes = predicted_classes[:, i]
            _true_positives = true_positives[:, i]
            _confidence_scores = confidence_scores

            if self.count_all_classes:
                classes = np.unique(np.concatenate([_gt_classes, _predicted_classes]))
            else:
                classes = np.unique(_gt_classes)

            # iterate over classes
            for c in classes:
                # get true positives and number of GT values
                tp = _true_positives[_predicted_classes == c]
                cs = _confidence_scores[_predicted_classes == c]
                ngt = np.sum(_gt_classes == c)

                # check if the list of TP is non empty
                if len(tp) > 0:
                    # remove invalid TP values and related confidence scores
                    valid = ~np.isnan(tp)
                    tp, cs = tp[valid], cs[valid]
                # if both TP and GT are non empty, then compute AP
                if len(tp) > 0 and ngt > 0:
                    prec, rec = self._compute_prec_rec(tp, cs, ngt)
                    if measure=='AP':
                        this_measure = self._compute_ap(prec, rec)
                    elif measure=='MR': #maximum recall
                        this_measure = self._compute_mr(prec, rec)
                    # turn into percentage
                    if self.percentages:
                        this_measure = this_measure * 100
                    # append to the list
                    measures.append(this_measure)
                # if both are empty, the AP is zero
                elif not (len(tp) == 0 and ngt == 0):
                    measures.append(0)
            # append the mAP value
            metrics.append(np.mean(measures))

        # return single value or list of values
        values = list(metrics)
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    @abstractmethod
    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT labels
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """

class ObjectOnlyMeanAveragePrecision(AbstractMeanAveragePrecision):
    def __init__(self, iou_threshold=0.5, top_k=3, count_all_classes=False):
        """
        Construct the object only mAP metric. This will compute the following metrics:
            - Box + Noun
            - Box
        :param iou_threshold:
        :param tti_threshold:
        :param top_k:
        :param count_all_classes:
        """
        super().__init__(2, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.names = ["Box + Noun mAP", "Box AP"]
        self.short_names = ["map_box_noun", "ap_box"]

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']
        boxes = np.ones(len(preds['nouns']))

        return np.vstack([
            nouns,  # box + noun, average over nouns
            boxes]  # box, just compute a single AP
        ).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)

        map_box_noun = boxes & nouns
        map_box = boxes
        # breakpoint()

        return np.vstack([map_box_noun, map_box]).T

class OverallMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP"""
    def __init__(self, iou_threshold=0.5, ttc_threshold=0.25, top_k=5, count_all_classes=False):
        """
        Construct the overall mAP metric. This will compute the following metrics:
            - Box AP
            - Box + Noun AP
            - Box + Verb AP
            - Box + TTC AP
            - Box + Verb + TTC AP
            - Box + Noun mAP
            - Box + Noun + Verb mAP
            - Box + Noun + TTC mAP
            - Box + Noun + Verb + TTC mAP
        :param iou_threshold: IOU threshold to check if a predicted box can be matched to a ground turth box
        :param ttc_threshold: TTC threshold to check if a predicted TTC is acceptable
        :param top_k: Top-K criterion for mAP. Discounts up to k-1 high scoring false positives
        :param count_all_classes: whether to also average across classes with no annotations. False is the default for many implementations.
        """
        super().__init__(12, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.tti_threshold = ttc_threshold

        self.names = ['Box AP',
                      'Box + Noun AP',
                      'Box + Verb AP',
                      'Box + TTC AP',
                      'Box + Noun + Verb AP',
                      'Box + Noun + TTC AP',
                      'Box + Verb + TTC AP',
                      'Box + Noun + Verb + TTC AP',
                      'Box + Noun mAP',
                      'Box + Noun + Verb mAP',
                      'Box + Noun + TTC mAP',
                      'Box + Noun + Verb + TTC mAP']

        self.short_names = ['ap_box',
                      'ap_box_noun',
                      'ap_box_verb',
                      'ap_box_ttc',
                      'ap_box_noun_verb',
                      'ap_box_noun_ttc',
                      'ap_box_verb_ttc',
                      'ap_box_noun_verb_ttc',
                      'map_box_noun',
                      'map_box_noun_verb',
                      'map_box_noun_ttc',
                      'map_box_noun_verb_ttc']

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']
        ones = np.ones(len(preds['nouns']))

        return np.vstack([
            ones, # ap_box - do not average
            ones, # ap_box_noun - do not average
            ones, # ap_box_verb - do not average
            ones, # ap_box_ttc - do not average
            ones, # ap_box_noun_verb - do not average
            ones, # ap_box_noun_ttc - do not average
            ones, # ap_box_verb_ttc - do not average
            ones, # ap_box_noun_verb_ttc - do not average
            nouns, # map_box_noun - average over nouns
            nouns, # map_box_noun_verb - average over nouns
            nouns, # map_box_noun_ttc - average over nouns
            nouns # map_box_noun_verb_ttc - average over nouns
        ]).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)
        verbs = (pred['verbs'] == gt_predictions['verbs'])
        ttcs = (np.abs(pred['ttcs'] - gt_predictions['ttcs']) <= self.tti_threshold)

        tp_box = boxes
        tp_box_noun = boxes & nouns
        tp_box_verb = boxes & verbs
        tp_box_ttc = boxes & ttcs
        tp_box_noun_verb = boxes & verbs & nouns
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_verb_ttc = boxes & verbs & ttcs
        tp_box_noun_verb_ttc = boxes & verbs & nouns & ttcs
        # breakpoint()

        return np.vstack([tp_box,  # ap_box
                          tp_box_noun,  # ap_box_noun
                          tp_box_verb,  # ap_box_verb
                          tp_box_ttc,  # ap_box_ttc
                          tp_box_noun_verb,  # ap_box_noun_verb
                          tp_box_noun_ttc,  # ap_box_noun_ttc
                          tp_box_verb_ttc,  # ap_box_verb_ttc
                          tp_box_noun_verb_ttc,  # ap_box_noun_verb_ttc
                          tp_box_noun,  # map_box_noun
                          tp_box_noun_verb,  # map_box_noun_verb
                          tp_box_noun_ttc,  # map_box_noun_ttc
                          tp_box_noun_verb_ttc  # map_box_noun_verb_ttc
                          ]).T

class STAMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP"""
    def __init__(self, iou_threshold=0.5, ttc_threshold=0.25, top_k=5, count_all_classes=False):
        """
        Construct the overall mAP metric. This will compute the following metrics:
            - Box + Noun mAP
            - Box + Noun + Verb mAP
            - Box + Noun + TTC mAP
            - Box + Noun + Verb + TTC mAP
        :param iou_threshold: IOU threshold to check if a predicted box can be matched to a ground turth box
        :param ttc_threshold: TTC threshold to check if a predicted TTC is acceptable
        :param top_k: Top-K criterion for mAP. Discounts up to k-1 high scoring false positives
        :param count_all_classes: whether to also average across classes with no annotations. False is the default for many implementations.
        """
        super().__init__(4, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.tti_threshold = ttc_threshold

        self.names = ['Box + Noun mAP',
                      'Box + Noun + Verb mAP',
                      'Box + Noun + TTC mAP',
                      'Box + Noun + Verb + TTC mAP']

        self.short_names = ['map_box_noun',
                            'map_box_noun_verb',
                            'map_box_noun_ttc',
                            'map_box_noun_verb_ttc']

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']

        return np.vstack([
            nouns, # map_box_noun - average over nouns
            nouns, # map_box_noun_verb - average over nouns
            nouns, # map_box_noun_ttc - average over nouns
            nouns # map_box_noun_verb_ttc - average over nouns
        ]).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)
        verbs = (pred['verbs'] == gt_predictions['verbs'])
        ttcs = (np.abs(pred['ttcs'] - gt_predictions['ttcs']) <= self.tti_threshold)

        tp_box_noun = boxes & nouns
        tp_box_noun_verb = boxes & verbs & nouns
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_noun_verb_ttc = boxes & verbs & nouns & ttcs
        # breakpoint()
        return np.vstack([tp_box_noun,  # map_box_noun
                          tp_box_noun_verb,  # map_box_noun_verb
                          tp_box_noun_ttc,  # map_box_noun_ttc
                          tp_box_noun_verb_ttc  # map_box_noun_verb_ttc
                          ]).T
