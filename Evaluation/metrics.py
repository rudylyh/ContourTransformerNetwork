import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import binary_dilation, disk
import cv2
import DataProvider.utils as DataUtils
from scipy.spatial.distance import directed_hausdorff


def hausdorff_dist(pred, gt):
    # dist = directed_hausdorff(pred, gt)[0]
    pred = DataUtils.sample_arc_point(pred, pred.shape[0])
    ctr_dist = cdist(pred, gt, 'euclidean')
    all_dist = ctr_dist.min(axis=1) # axis=1: the min dist from each pred point to all gt points
    ave_dist = all_dist.mean()
    max_dist = all_dist.max()
    return ave_dist, max_dist


### The result is same with hausdorff_dist()
def hausdorff_dist_v2(pred, gt):
    # dist = directed_hausdorff(pred, gt)[0]
    pred = DataUtils.sample_arc_point(pred, pred.shape[0])
    hausdorff_dist = max(directed_hausdorff(pred, gt)[0], directed_hausdorff(gt, pred)[0])
    return hausdorff_dist


def get_ctr_iou(pred, gt, h, w):
    pred_img = np.zeros((h, w), dtype=np.uint8)
    gt_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(pred_img, [pred], 1)
    cv2.fillPoly(gt_img, [gt], 1)
    intersection = np.sum(np.logical_and(pred_img == 1, gt_img == 1))
    union = np.sum(np.logical_or(pred_img == 1, gt_img == 1))
    # pred_img[pred_img==1] = 255
    # gt_img[gt_img==1] = 255
    # cv2.imshow('1', pred_img)
    # cv2.imshow('2', gt_img)
    # cv2.waitKey()
    # print(float(intersection)/union)
    return float(intersection)/union


def curve_f1_score(h, w, pred, gt, bound_pix=1):
    img_pred = np.zeros((h, w))
    img_gt = np.zeros((h, w))
    img_pred[pred[:, 1], pred[:, 0]] = 1
    img_gt[gt[:, 1], gt[:, 0]] = 1

    fg_dil = binary_dilation(img_pred, disk(bound_pix))
    gt_dil = binary_dilation(img_gt, disk(bound_pix))

    # Get the intersection
    gt_match = img_gt * fg_dil
    fg_match = img_pred * gt_dil

    # Area of the intersection
    n_fg = np.sum(img_pred)
    n_gt = np.sum(img_gt)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)

    return F1
