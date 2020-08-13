import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import torch
import pdb
from bbox import bbox_overlaps

def OICRLayer(boxes, cls_prob, im_labels, cfg_TRAIN_FG_THRESH = 0.5):
    # boxes = boxes[...,1:]
    # boxes:[1, 3189, 4]
    # 上一层传来的cls_prob:[1, 3819, 20/21]
    # image level label -> im_labels:[1, 20]
    proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels)
    labels, cls_loss_weights = _sample_rois(boxes, proposals, 21)
    return labels, cls_loss_weights

def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # 图片labels
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

        # 裁剪掉背景分类
    if 21 == cls_prob.shape[2] : # added 1016
        cls_prob = cls_prob[:, :, 1:]

    # 统计GT类得分最高的box
    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:,:, i].data.cpu()
            max_index = np.argmax(cls_prob_tmp)
            # m = boxes[:,max_index, :].reshape(1, -1).cpu()
            gt_boxes = np.vstack((gt_boxes, boxes[:,max_index, :].reshape(1, -1).cpu()))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
            gt_scores = np.vstack((gt_scores,
                cls_prob_tmp[:, max_index] ))  # * np.ones((1, 1), dtype=np.float32)))
            cls_prob[:, max_index, :] = 0 #in-place operation <- OICR code but I do not agree
    # proposals {[1, 4], [1, 1], [1, 1]}
    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}
    return proposals


def _sample_rois(all_rois, proposals, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[0].cpu(), dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    try :
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
    except :
        pdb.set_trace()

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < 0.5)[0]

    labels[bg_inds] = 0
    real_labels = np.zeros((labels.shape[0], 21))
    for i in range(labels.shape[0]) :
        real_labels[i, labels[i]] = 1
    rois = all_rois.cpu()
    return real_labels, cls_loss_weights
