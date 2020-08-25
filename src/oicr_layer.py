import torch
from utils import *

def oicr_algorithm(xr0, gt_label, regions, K=3):
    R = regions.size()[1] # R
    # then do the online instance classifier refinement
    wrk_list = torch.zeros((K, R)).to(cfg.DEVICE)
    # R x 21 x k
    yrk_list = torch.zeros((K, R, (1 + len(VOC_CLASSES))))
    yrk_list[:, :, -1] = 1.0
    yrk_list = yrk_list.to(cfg.DEVICE)
#     # here is just to calculate the supervised information 
#     # do not need grad any more
    with torch.no_grad():
        for k in range(K):
            wrk = wrk_list[k, :]
            yrk = yrk_list[k, :, :]
            IoUs = torch.full((R, ), - np.inf).to(cfg.DEVICE)
            for c in range(len(VOC_CLASSES)):
                if gt_label[0][c] == 1.0:
                    top_id = torch.argmax(xr0[k][:, c])
                    top_score = xr0[k][top_id][c]
#                     writer.add_scalar("top_score", top_score, 0)
#                     print(top_score)
                    top_box = regions[0][top_id:top_id+1]
                    IoUs_temp = one2allbox_iou(top_box, regions[0])
                    IoU_mask = torch.where(IoUs_temp > IoUs)
                    IoUs[IoU_mask] = IoUs_temp[IoU_mask]
                    wrk[IoU_mask] = top_score
#                     y_mask = torch.where(IoUs > cfg.TRAIN.It)
                    y_mask = torch.where(IoUs[IoU_mask] > cfg.TRAIN.It)
                    yrk[y_mask] = 0.0
                    yrk[y_mask] += torch.eye(1 + len(VOC_CLASSES))[c].to(cfg.DEVICE)
    return wrk_list, yrk_list
