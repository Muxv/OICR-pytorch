import torch
import numpy as np
import cv2

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
