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
from models import OICR_Alexnet, OICR_VGG16
from refine_loss import WeightedRefineLoss
from datasets import VOCDectectionDataset


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
        oicr = OICR_Alexnet()
    elif pretrained == 'vgg16':
        oicr = OICR_VGG16
    oicr.init_model()
    oicr.to(cfg.DEVICE)
    oicr.train()
    
    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, cfg.TRAIN.BATCH_SIZE, shuffle=True)

    optimizer = optim.SGD(oicr.parameters(),
                          lr=cfg.TRAIN.LR,
                          weight_decay=cfg.TRAIN.WD,
                          momentum=cfg.TRAIN.MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 28], gamma=0.1)
    log_file = cfg.PATH.LOG_PATH + f"oicr_{pretrained}_" +  datetime.datetime.now().strftime('%m-%d_%H:%M') + ".txt"
    N = len(train_loader)
    bceloss = nn.BCELoss(reduction="sum")
    refineloss = WeightedRefineLoss()
    
    for epoch in tqdm(range(cfg.TRAIN.EPOCH_07), "Total"):
        iter_id = 0 # use to do accumulated gd
        epoch_loss = 0.0
        for img, gt_box, gt_label, regions in tqdm(train_loader, f"Epoch {epoch}"):
            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_label = gt_label.to(cfg.DEVICE) # 1, C
            refine_scores = []

            proposal_scores, refine_scores = oicr(img, regions)
            
                    
            xr0 = torch.zeros((R, 21)).to(cfg.DEVICE) # xj0
            xr0[:, :20] = proposal_scores
            # R+1 x 21
            refine_scores.insert(0, xr0)
            
            
            cls_scores = torch.sum(proposal_scores, dim=0)
            cls_scores = torch.clamp(cls_scores, min=0.0, max=1.0)
            b_loss = bceloss(cls_scores, gt_label[0])
            r_loss = [None for _ in range(cfg.K)]

            # then do the online instance classifier refinement
            # 
            wrk_list = torch.Tensor([[0 for _ in range(R)] for _ in range(cfg.K)]).to(cfg.DEVICE)
            yrk_list = torch.Tensor([
                [[0 for _ in range(1 + len(VOC_CLASSES))]for _ in range(R)]for _ in range(cfg.K)
            ])
            yrk_list = yrk_list.to(cfg.DEVICE)
            yrk_list[:, :, -1] = 1.0

            for k in range(cfg.K):
                IoUs = torch.Tensor([-np.inf for _ in range(R)]).to(cfg.DEVICE)
                for c in range(len(VOC_CLASSES)):
                    if gt_label[0][c] == 1:
                        top_id = torch.argmax(refine_scores[k][:, c])
                        top_box = regions[0][top_id:top_id+1]
                        IoUs_temp = one2allbox_iou(top_box, regions[0])
                        IoU_mask = IoUs_temp > IoUs
                        IoUs = IoU_mask * IoUs_temp
                        wrk_list[k] = IoU_mask * refine_scores[k][top_id][c]
                        y_mask = IoUs > cfg.TRAIN.It
                        yrk_list[k][y_mask] = 0.0
                        yrk_list[k][y_mask] += torch.Tensor([1 if _ == c else 0 for _ in range(21)]).to(cfg.DEVICE)
                r_loss[k] = refineloss(refine_scores[k+1], 
                                       yrk_list[k].clone(),
                                       wrk_list[k].clone())

            loss = b_loss + sum(r_loss)
            epoch_loss += loss.item()
            loss.backward()
            iter_id += 1
            if iter_id % cfg.TRAIN.ITER_SIZE == 0 or iter_id == N:
                optimizer.step()
                optimizer.zero_grad()


        print(f"Epoch {epoch} Loss is {epoch_loss/N}")
        write_log(log_file, f"Epoch {epoch} Loss is {epoch_loss/N}")         
        scheduler.step()
        if epoch % 5 == 0:
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : oicr.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss': loss
                }, cfg.PATH.PT_PATH + f"oicr_{year}_{pretrained}_Epoch {epoch}.pt")
    torch.save({
        'model_state_dict' : oicr.state_dict(),
    }, cfg.PATH.PT_PATH + f"oicr_{year}_{pretrained}_Epoch finished.pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
    
    