import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import datetime

from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import optim
from torchvision.ops import roi_pool, nms
from sklearn.metrics import average_precision_score
from config import cfg
from utils import *
from models import OICR, MIDN_Alexnet, MIDN_VGG16
from refine_loss import WeightedRefineLoss
from datasets import VOCDectectionDataset

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
    writer = SummaryWriter("../boardX")
    args = parse.parse_args()
    year = args.year
    pretrained = args.pretrained
    oicr = None
    midn = None
    check_points = True
    start_epoch = 0

    if pretrained == 'alexnet':
        midn = MIDN_Alexnet()
    elif pretrained == 'vgg16':
        midn = MIDN_VGG16()
    midn.init_model()
    midn.to(cfg.DEVICE)

    oicr = OICR(cfg.K)
    oicr.init_model()
    oicr.to(cfg.DEVICE)

    if check_points:
        checkpoints = torch.load(cfg.PATH.PT_PATH + "Model_2007_alexnet_Epoch finished.pt")
        start_epoch = 10
        midn.load_state_dict(checkpoints['midn_model_state_dict'])


    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, cfg.TRAIN.BATCH_SIZE, shuffle=False)

    midn_optimizer = optim.Adam(midn.parameters(),
                               lr=cfg.TRAIN.LR,
                               weight_decay=cfg.TRAIN.WD)
    scheduler = optim.lr_scheduler.MultiStepLR(midn_optimizer,
                                               milestones=[cfg.TRAIN.LR_STEP,
                                                           cfg.TRAIN.EPOCH],
                                               gamma=cfg.TRAIN.LR_MUL)
    oicr_optimizer = optim.Adam(oicr.parameters(),
                                lr=0.1 * cfg.TRAIN.LR,
                                weight_decay=cfg.TRAIN.WD)

    log_file = cfg.PATH.LOG_PATH + f"oicr_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M')+ ".txt"
    N = len(train_loader)
    bceloss = nn.BCELoss(reduction="sum")
    refineloss = WeightedRefineLoss()



    midn.train()
    oicr.train()
    
    img_num = 0
    for epoch in tqdm(range(start_epoch, cfg.TRAIN.EPOCH + 1), "Total"):
    #     img_num = 0
        iter_id = 0 # use to do accumulated gd
        epoch_b_loss = 0.0
        epoch_r_loss = 0.0
        for img, gt_box, gt_label, regions in tqdm(train_loader, f"Epoch {epoch}"):

            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_label = gt_label.to(cfg.DEVICE) # 1, C

            with torch.no_grad():
                fc7, proposal_scores = midn(img, regions)
                cls_scores = torch.sum(proposal_scores, dim=0)
                cls_scores = torch.clamp(cls_scores, min=0.0, max=1.0)
                b_loss = bceloss(cls_scores, gt_label[0])
            if epoch < 10:
                b_loss.backward()
                epoch_b_loss += b_loss.item()

            else:
                refine_scores = oicr(fc7.detach())
                xr0 = torch.zeros((R, 21)).to(cfg.DEVICE) # xj0
                xr0[:, :20] = proposal_scores.detach()
                # R+1 x 21
                refine_scores.insert(0, xr0)

                r_loss = [None for _ in range(cfg.K)]
                wrk_list, yrk_list = oicr_algorithm(refine_scores, gt_label, regions, cfg.K)
                for k in range(cfg.K):
                    r_loss[k] = refineloss(refine_scores[k+1], 
                                           yrk_list[k],
                                           wrk_list[k])
                writer.add_histogram("rscore1", refine_scores[1].cpu().detach().numpy(), img_num)

    #             b_loss.backward()
                sum(r_loss).backward()
    #             epoch_b_loss += b_loss.item()
                epoch_r_loss += sum(r_loss).item()
            img_num += 1
            iter_id += 1
            if iter_id % cfg.TRAIN.ITER_SIZE == 0 or iter_id == N:
    #             midn_optimizer.step()
    #             midn_optimizer.zero_grad()
                if epoch >= 10:
                    oicr_optimizer.step()
                    oicr_optimizer.zero_grad()

        if epoch < 10:
            print(f"Epoch {epoch} b_Loss is {epoch_b_loss/N}")
    #         write_log(log_file, f"Epoch {epoch} b_Loss is {epoch_b_loss/N}")
        else:
            print(f"Epoch {epoch} b_Loss is {epoch_b_loss/N}")
    #         write_log(log_file, f"Epoch {epoch} b_Loss is {epoch_b_loss/N}") 
            print(f"Epoch {epoch} r_Loss is {epoch_r_loss/N}")
    #         write_log(log_file, f"Epoch {epoch} r_Loss is {epoch_r_loss/N}")
    #     break

    #     scheduler.step()
        torch.save({
            'epoch' : epoch,
            'oicr_model_state_dict' : oicr.state_dict(),
            'oicr_optimizer_state_dict' : oicr_optimizer.state_dict(),
            'r_loss': r_loss
            }, cfg.PATH.PT_PATH + f"OICR_{year}_{pretrained}_Epoch {epoch}.pt")