import torch
from torchvision.ops import nms
from utils import *

def oicr_crs_algorithm(xrk_list, gt_count, regions, K=3, T=0.1, max_k=3):
    R = regions.size()[1] # R
    # then do the online instance classifier refinement
    wrk_list = torch.zeros((K, R)).to(cfg.DEVICE)
    # R x 21 x k
    yrk_list = torch.zeros((K, R, (1 + len(VOC_CLASSES))))
    yrk_list[:, :, -1] = 1.0
    yrk_list = yrk_list.to(cfg.DEVICE)

    with torch.no_grad():
        for k in range(K):
            xrk = xrk_list[k]
            wrk = wrk_list[k, :]
            yrk = yrk_list[k, :, :]
            IoUs = torch.full((R, ), - np.inf).to(cfg.DEVICE)
            
            count_g = crs(xrk, gt_count, regions, T, max_k)

            for c, group in count_g.items():
                for top_id in group:
                    top_score = xrk[top_id][c]
                    top_box = regions[0][top_id:top_id+1]
                    IoUs_temp = one2allbox_iou(top_box, regions[0])
                    IoU_mask = torch.where(IoUs_temp > IoUs)
                    IoUs[IoU_mask] = IoUs_temp[IoU_mask]
                    wrk[IoU_mask] = top_score
                    y_mask = torch.where(IoUs[IoU_mask] > cfg.TRAIN.It)
                    yrk[y_mask] = 0.0
                    yrk[y_mask] += torch.eye(1 + len(VOC_CLASSES))[c].to(cfg.DEVICE)
    return wrk_list, yrk_list


def oicr_algorithm(xrk_list, gt_label, regions, K=3):
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
                if gt_label[0][c] >= 1.0:
                    top_id = torch.argmax(xrk_list[k][:, c])
                    top_score = xrk_list[k][top_id][c]
                    top_box = regions[0][top_id:top_id+1]
                    IoUs_temp = one2allbox_iou(top_box, regions[0])
                    IoU_mask = torch.where(IoUs_temp > IoUs)
                    IoUs[IoU_mask] = IoUs_temp[IoU_mask]
                    wrk[IoU_mask] = top_score
                    y_mask = torch.where(IoUs[IoU_mask] > cfg.TRAIN.It)
                    yrk[y_mask] = 0.0
                    yrk[y_mask] += torch.eye(1 + len(VOC_CLASSES))[c].to(cfg.DEVICE)
    return wrk_list, yrk_list


def check_crs_iou(G, new_box, regions, T=0.1):
    for i in G:
        if crs_iou(regions[0][i], regions[0][new_box]) >= T:
            return False
    return True


def crs(xrk, gt_count, regions, T=0.1, max_k=3, IoU=0.3):
    """
    include nms + CRS
    """
#     N = regions.size()[1]
    s = 0 
    G = [] # store roi's index
    T = 0.1

    Count_G = {}
    for c in range(len(VOC_CLASSES)):
        if gt_count[0][c] >= 1.0:
            p = xrk[:, c]
            rk = regions[0]
            
            nms_filter = nms(rk, p, IoU)
            

            
            count = int(gt_count[0][c])
            count = min(count, max_k)
                        # nms 
            # ----------------------------------
            
            sort_p = p[nms_filter]
            indices = nms_filter
            N = nms_filter.size()[0]
            # ----------------------------------
#             sort_p, indices = torch.sort(p, descending=True)

            G_best = []
            s_max = 0
            # this case run so fast
            if count == 1:
                # choose the best box
                G_best = [indices[0].item()]
                s_max = sort_p[0]
            else:
                for i in range(N):
                    G = [indices[i].item()]
                    s = sort_p[i]
                    
                    if sort_p[i] * count < s_max:
                        break
                        
                    for j in range(i + 1, N):
                        new_box_id = indices[j].item()
                        if check_crs_iou(G, new_box_id, regions, T):
                            G.append(new_box_id)
                            s += sort_p[j]
                            if len(G) == count or j == N - 1:
                                if s > s_max:
                                    G_best = G
                                    s_max = s
                                break
            Count_G[c] = G_best
    return Count_G