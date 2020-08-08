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
from models import *
from refine_loss import WeightedRefineLoss
from datasets import VOCDectectionDataset
from tensorboardX import SummaryWriter

writer = SummaryWriter('./runs')   # 数据存放在这个文件夹


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
        "--pretrained", type=str, default='vgg16', help="which pretrained model to use"
    )

    

    args = parse.parse_args()
    year = args.year
    pretrained = args.pretrained
    oicr = None
    midn = None
    
    if pretrained == 'alexnet':
        model = Combined_Alexnet(cfg.K)

    if pretrained == 'vgg16':
        model = Combined_VGG16(cfg.K)


    model.to(cfg.DEVICE)
    model.init_model()
    
    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, cfg.TRAIN.BATCH_SIZE, shuffle=True)
    train_iterator = iter(train_loader)

    
    MAX_ITER_SIZE = 50000
    STEP_ITER = 35000
    WARM_ITER = 300
    WARM_FACTOR = 1.0/3.0
    BASE_LR= 1e-3
    GAMMA = 0.1
    PRINT_STEP = 1000
    
    
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
                
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.TRAIN.WD},
        {'params': bias_params,
         'lr': 0 * (cfg.TRAIN.BIAS_DOUBLE_LR + 1),
         'weight_decay':  0},
    ]
    
    optimizer = optim.SGD(params,
                          momentum=cfg.TRAIN.MOMENTUM)
    
    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.


    log_file = cfg.PATH.LOG_PATH + f"Model_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M')+ ".txt"
    N = len(train_loader)
    bceloss = nn.BCELoss(reduction="sum")
    refineloss = WeightedRefineLoss()
    model.train()
    step_B_loss = 0
    step_R_loss = 0
    for step in tqdm(range(1, MAX_ITER_SIZE+1), "Total Iter"):

        # warm up
        if step < WARM_ITER:
            alpha = step / WARM_ITER
            warmup_factor = WARM_FACTOR * (1 - alpha) + alpha
            lr_new = BASE_LR * warmup_factor
            update_learning_rate(optimizer, lr, lr_new)
            lr = optimizer.param_groups[0]['lr']
            assert lr == lr_new
        elif step == WARM_ITER:
            lr_new = BASE_LR
            update_learning_rate(optimizer, lr, lr_new)
            lr = optimizer.param_groups[0]['lr']
            assert lr == lr_new
        elif step == STEP_ITER:
            lr_new = BASE_LR * GAMMA
            update_learning_rate(optimizer, lr, lr_new)
            lr = optimizer.param_groups[0]['lr']
            assert lr == lr_new  
        optimizer.zero_grad()
        for inner_iter in range(4):
            try:
                img, gt_box, gt_label, regions = next(train_iterator)
            except StopIteration:
                img, gt_box, gt_label, regions = next(train_iterator)
                train_iterator = next(train_loader)
            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_label = gt_label.to(cfg.DEVICE) # 1, C

            
            refine_scores, proposal_scores = model(img, regions)
            cls_scores = torch.sum(proposal_scores, dim=0)
            cls_scores = torch.clamp(cls_scores, min=0, max=1)
            b_loss = bceloss(cls_scores, gt_label[0])
            
#             writer.add_histogram('cls_scores', cls_scores.detach().cpu(), 0)
#             writer.add_histogram('refine_scores0', refine_scores[0].detach().cpu(), 0)
#             writer.add_histogram('refine_scores1', refine_scores[1].detach().cpu(), 0)
#             writer.add_histogram('refine_scores2', refine_scores[2].detach().cpu(), 0)
            
            xr0 = torch.zeros((R, 21)).to(cfg.DEVICE) # xj0
            xr0[:, :20] = proposal_scores.detach()
                # R+1 x 21
            xrk_list = [_.clone().detach() for _ in refine_scores]
            xrk_list.insert(0, xr0)

            r_loss = [None for _ in range(cfg.K)]
            wrk_list, yrk_list = oicr_algorithm(xrk_list, gt_label, regions, cfg.K)

            for k in range(cfg.K):
#                 r_scores = torch.clamp(refine_scores[k], min=1e-6, max=1-1e-6)
                r_loss[k] = refineloss(refine_scores[k], 
                                       yrk_list[k],
                                       wrk_list[k])
            step_B_loss += b_loss.item()
            step_R_loss += sum(r_loss).item()
            loss = b_loss + sum(r_loss)
            loss.backward()
            
        optimizer.step()
        if step % PRINT_STEP == 0:
            print(f"STEP {step} b_Loss is {step_B_loss/PRINT_STEP}")
            write_log(log_file, f"STEP {step} b_Loss is {step_B_loss/PRINT_STEP}")
            print(f"STEP {step} r_Loss is {step_R_loss/PRINT_STEP}")
            write_log(log_file, f"STEP {step} r_Loss is {step_R_loss/PRINT_STEP}")
            step_B_loss = 0
            step_R_loss = 0
#             # disk space is not enough
            if (os.path.exists(cfg.PATH.PT_PATH + f"Model_{year}_{pretrained}_{step-PRINT_STEP}.pt")):
                os.remove(cfg.PATH.PT_PATH + f"Model_{year}_{pretrained}_{step-PRINT_STEP}.pt")
            torch.save({
               'whole_model_state_dict' : model.state_dict(),
               }, cfg.PATH.PT_PATH + f"WholeModel_{year}_{pretrained}_{step-PRINT_STEP}.pt")

    torch.save({
                'whole_model_state_dict' : model.state_dict(),
                }, cfg.PATH.PT_PATH + f"[FS]WholeModel_{year}_{pretrained}_{step-PRINT_STEP}.pt")
            
    

    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
    
    
