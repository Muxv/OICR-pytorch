import torch
import numpy as np
import cv2
from tqdm import tqdm
from chainercv.evaluations import eval_detection_voc
from config import cfg
from torchvision.ops import roi_pool, nms

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def record_info(s, log_file):
    print(s)
    write_log(log_file, s)
    
def draw_box(img, boxes):
    """
    img : PIL Image
    boxes: np.darray shape (N, 4)
    """
    p = np.asarray(img)
    for box in boxes:
        cv2.rectangle(p, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
    plt.imshow(p)
    
def get_model_name(propose_way, year, model_name):
    name = ""
    if propose_way == "selective_search":
        name += "ssw_"
    else:
        name += "eb_"
    
    name += str(year)+ "_" 
    name += model_name

    return name

def one2allbox_iou(target_box, others):
    """
     calculate the iou of box A to list of boxes
     target_box : Tensor()  1 * 4 
     others : Tensor()      N * 4 
     return  N * 1  ...iou
    
    """

    # get the min of xmax and ymax which organize the Intersection
    max_xy = torch.min(target_box[:, 2:], others[:, 2:]) 
    min_xy = torch.max(target_box[:, :2], others[:, :2])
    # get the xdistance and y distance
    inter_wh = torch.clamp((max_xy - min_xy + 1), min=0)
    I = inter_wh[:, 0] * inter_wh[:, 1]
    A = (target_box[:, 2] - target_box[:, 0] + 1) * (target_box[:, 3] - target_box[:, 1] + 1)
    B = (others[:, 2] - others[:, 0] + 1) * (others[:, 3] - others[:, 1] + 1)
    return I / (A + B - I)


def write_log(path, content):
    with open(path, 'a') as f:
        f.write(content + "\n")
        
def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio
        
        
def update_learning_rate(optimizer, cur_lr, new_lr, bias_double_lr=True):
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > 1.1:
            print(f'Changing learning rate {cur_lr} -> {new_lr}')
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if ind == 1 and bias_double_lr:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            param_keys += param_group['params']

def evaluate(model, testdl, log):
    total_pred_boxes = []
    total_pred_labels = []
    total_pred_scores = []
    total_true_boxes = []
    total_true_labels = []
    k = cfg.K
    with torch.no_grad():
        model.eval()
        for n_imgs, gt, n_regions, region in tqdm(testdl, "Evaluation"):
            region = region.to(cfg.DEVICE)
            avg_scores = torch.zeros((len(region[0]), 20), dtype=torch.float32)
           
            per_img = n_imgs[0].to(cfg.DEVICE)
            per_region = n_regions[0].to(cfg.DEVICE)
            ref_scores1, ref_scores2, ref_scores3, proposal_scores = model(per_img, per_region)
            avg_scores += (ref_scores1 + ref_scores2 + ref_scores3)[:, 1:].detach().cpu() / k

            gt = gt.numpy()[0]
            gt_boxex = gt[:, :4]
            gt_labels = gt[:, -1]

            gt_labels_onehot = np.zeros(20)
            for label in gt_labels:
                gt_labels_onehot[int(label)] = 1

            per_pred_boxes = []
            per_pred_scores = []
            per_pred_labels = []

            region = region[0].cpu()

            for i in range(20):
                cls_scores = avg_scores[:, i]
                cls_region = region
                nms_filter = nms(cls_region, cls_scores, 0.3)
                per_pred_boxes.append(cls_region[nms_filter].numpy())
                per_pred_scores.append(cls_scores[nms_filter].numpy())
                per_pred_labels.append(np.full(len(nms_filter), i, dtype=np.int32))

            total_pred_boxes.append(np.concatenate(per_pred_boxes, axis=0))
            total_pred_scores.append(np.concatenate(per_pred_scores, axis=0))
            total_pred_labels.append(np.concatenate(per_pred_labels, axis=0))
            total_true_boxes.append(gt_boxex)
            total_true_labels.append(gt_labels)

        result = eval_detection_voc(
            total_pred_boxes,
            total_pred_labels,
            total_pred_scores,
            total_true_boxes,
            total_true_labels,
            iou_thresh=0.5,
            use_07_metric=True,
        )
        print(f"Avg AP: {result['ap']}")
        print(f"Avg mAP: {result['map']}")
        write_log(log, f"Avg AP: {result['ap']}")
        write_log(log, f"Avg mAP: {result['map']}")
        return result['map']