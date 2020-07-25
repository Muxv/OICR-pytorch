import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import datetime

from tqdm import tqdm
from torch import optim
from torchvision.ops import roi_pool, nms
from sklearn.metrics import average_precision_score
from config import cfg
from utils import *
from models import OICR, MIDN_Alexnet, MIDN_VGG16
from refine_loss import WeightedRefineLoss
from datasets import VOCDectectionDataset
from sklearn.metrics import average_precision_score


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def oicr_algorithm(refined_scores, gt_label, regions, K=3):
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
                    top_id = torch.argmax(refine_scores[k][:, c])
                    top_box = regions[0][top_id:top_id+1]
                    IoUs_temp = one2allbox_iou(top_box, regions[0])
                    IoU_mask = torch.where(IoUs_temp > IoUs)
                    IoUs[IoU_mask] = IoUs_temp[IoU_mask]
                    wrk[IoU_mask] = refine_scores[k][top_id][c]
                    y_mask = torch.where(IoUs > cfg.TRAIN.It)
                    yrk[y_mask] = 0.0
                    yrk[y_mask] += torch.eye(1 + len(VOC_CLASSES))[c].to(cfg.DEVICE)
    return wrk_list, yrk_list


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train OICR")
    parse.add_argument(
        "--year", type=str, default='2007', help="Use which year of VOC"
    )
    parse.add_argument(
        "--pretrained", type=str, default='alexnet', help="which pretrained model to use"
    )


    args = parse.parse_args()
    year = args.year
    pretrained = args.pretrained
    oicr = None
    if pretrained == 'alexnet':
        midn = MIDN_Alexnet()
        lr = cfg.TRAIN.LR
        lr_step = cfg.TRAIN.LR_STEP
        lr_oicr = cfg.TRAIN.OICR_LR
        epochs = cfg.TRAIN.EPOCH
    elif pretrained == 'vgg16':
        midn = MIDN_VGG16()
        lr = cfg.TRAIN.VGG_LR
        lr_step = cfg.TRAIN.VGG_LR_STEP
        lr_oicr = cfg.TRAIN.OICR_LR
        epochs = cfg.TRAIN.VGG_EPOCH
    midn.init_model()
    midn.to(cfg.DEVICE)
    midn.train()
    
    
    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, cfg.TRAIN.BATCH_SIZE, shuffle=True)

    midn_optimizer = optim.Adam(midn.parameters(),
                               lr=lr,
                               weight_decay=cfg.TRAIN.WD)
    midn_scheduler = optim.lr_scheduler.MultiStepLR(midn_optimizer,
                                                    milestones=[lr_step,
                                                                epochs + 1],
                                                    gamma=cfg.TRAIN.LR_MUL)
#     oicr_optimizer = optim.Adam(oicr.parameters(),
#                                 lr=lr_oicr,
#                                 weight_decay=cfg.TRAIN.WD)

    log_file = cfg.PATH.LOG_PATH + f"midn_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M')+ ".txt"
    N = len(train_loader)
    bceloss = nn.BCELoss(reduction="sum")
#     refineloss = WeightedRefineLoss()s
    midn.train()
    
    for epoch in tqdm(range(1, epochs+1), "Total"):
        iter_id = 0 # use to do accumulated gd
        y_pred = []
        y_true = []

        epoch_b_loss = 0.0
        for img, gt_box, gt_label, regions in tqdm(train_loader, f"Epoch {epoch}"):
            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_label = gt_label.to(cfg.DEVICE) # 1, C

            fc7, proposal_scores = midn(img, regions)
            cls_scores = torch.sum(proposal_scores, dim=0)
            cls_scores = torch.clamp(cls_scores, min=0.0, max=1.0)
            b_loss = bceloss(cls_scores, gt_label[0])
            b_loss.backward()
            epoch_b_loss += b_loss.item()

            y_pred.append(cls_scores.detach().cpu().numpy().tolist())
            y_true.append(gt_label[0].detach().cpu().numpy().tolist())
            iter_id += 1
            if iter_id % cfg.TRAIN.ITER_SIZE == 0 or iter_id == N:
                midn_optimizer.step()
                midn_optimizer.zero_grad()

        cls_ap = []
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        for i in range(20):
            cls_ap.append(average_precision_score(y_true[:,i], y_pred[:,i])) 
        print(f"Epoch {epoch} classify AP is {str(cls_ap)}")
        write_log(log_file, f"Epoch {epoch} classify AP is {str(cls_ap)}")

        print(f"Epoch {epoch} classify mAP is {str(sum(cls_ap)/20)}")
        write_log(log_file, f"Epoch {epoch} classify mAP is {str(sum(cls_ap)/20)}")
        
        print(f"Epoch {epoch} b_Loss is {epoch_b_loss/N}")
        write_log(log_file, f"Epoch {epoch} b_Loss is {epoch_b_loss/N}")
        
        print("-" * 30)
        write_log(log_file, "-" * 30)
    
        if epoch % save_step == 0:
            # disk space is not enough
            if (os.path.exists(cfg.PATH.PT_PATH + f"Model_{year}_{pretrained}_{epoch-save_step}.pt")):
                os.remove(cfg.PATH.PT_PATH + f"Model_{year}_{pretrained}_{epoch-save_step}.pt")
            torch.save({
               'midn_model_state_dict' : midn.state_dict(),
               }, cfg.PATH.PT_PATH + f"Model_{year}_{pretrained}_{epoch}.pt")

    torch.save({
                'midn_model_state_dict' : midn.state_dict(),
                }, cfg.PATH.PT_PATH + f"[F]Model_{year}_{pretrained}.pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
    
    