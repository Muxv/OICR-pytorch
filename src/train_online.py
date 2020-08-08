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

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# writer = SummaryWriter('./runs')
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
    parse.add_argument(
        "--comment",  type=str, default='', help="add comment"
    )


    save_step = 5
    args = parse.parse_args()
    year = args.year
    pretrained = args.pretrained
    comment = args.comment
    oicr = None
    midn = None
    
    if pretrained == 'alexnet':
        model = Combined_Alexnet(cfg.K, cfg.Groups)
    if pretrained == 'vgg16':
        model = Combined_VGG16(cfg.K, cfg.Groups)
        
#     lr = cfg.TRAIN.LR
    eva_th = 3
    lr = 1e-3
    lr_step = 6
#     epochs = cfg.TRAIN.EPOCH
    epochs = 21
    start_epoch = 9
    
    
    log_file = cfg.PATH.LOG_PATH + f"Model_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M')+ ".txt"
    record_info(f"Full Epoch {epochs}", log_file)
    record_info(f"Base LR {lr}", log_file)
    record_info("-" * 30, log_file)

    model.to(cfg.DEVICE)
    model.init_model()

    checkpoints = torch.load(cfg.PATH.PT_PATH + "BestModel_2007_vgg16_8.pt")
    model.load_state_dict(checkpoints['whole_model_state_dict'])

    
    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, cfg.TRAIN.BATCH_SIZE, shuffle=True)
    testdata = VOCDectectionDataset("~/data/", year, 'test')
    test_loader = data.DataLoader(testdata, 1, shuffle=False)

#     optimizer = optim.Adam(model.parameters(),
#                            lr=lr,
#                            weight_decay=cfg.TRAIN.WD)
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
         'lr': lr,
         'weight_decay': cfg.TRAIN.WD},
        {'params': bias_params,
         'lr': lr * (cfg.TRAIN.BIAS_DOUBLE_LR + 1),
         'weight_decay':  0},
    ]
    
    optimizer = optim.SGD(params,
                          momentum=cfg.TRAIN.MOMENTUM)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[lr_step,
                                                           epochs],
                                               gamma=cfg.TRAIN.LR_MUL)
    iter_id = 0
    mAP = 0
    best_mAP = 0
    best_model = None
    best_epoch = 0

    N = len(train_loader)
    bceloss = nn.BCELoss(reduction="sum")
    refineloss = WeightedRefineLoss()
    model.train()
    
    
    for epoch in tqdm(range(start_epoch, epochs+1), "Total"):
        epoch_b_loss = 0.0
        epoch_r_loss = 0.0
        for img, gt_box, gt_label, regions in tqdm(train_loader, f"Epoch {epoch}"):
            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_label = gt_label.to(cfg.DEVICE) # 1, C
            
            refine_scores, proposal_scores = model(img, regions)
            cls_scores = torch.sum(proposal_scores, dim=0)
            cls_scores = torch.clamp(cls_scores, min=0, max=1)
            
            b_loss = bceloss(cls_scores, gt_label[0])
            epoch_b_loss += b_loss.item()


            xr0 = torch.zeros((R, 21)).to(cfg.DEVICE) # xj0
            xr0[:, :20] = proposal_scores.detach()
                # R+1 x 21
            xrk_list = [_.clone().detach() for _ in refine_scores]
            xrk_list.insert(0, xr0)

            r_loss = [None for _ in range(cfg.K)]
            wrk_list, yrk_list = oicr_algorithm(xrk_list, gt_label, regions, cfg.K)
#             wrk_list = wrk_list * 10
#             print(wrk_list)
#             wrk_list = wrk_list * 10 + 1
#             writer.add_histogram('wrk_list_0', wrk_list[0], iter_id)
#             writer.add_histogram('refine_scores0', refine_scores[0], iter_id)
#             true_labels = torch.where(yrk_list[0] == 1.0)
#             writer.add_histogram('trueX_0', refine_scores[0][true_labels], iter_id)
#             false_labels = torch.where(yrk_list[0] != 1.0)
#             writer.add_histogram('FalseX_0', refine_scores[0][false_labels], iter_id)
            
#             writer.add_histogram('wrk_list_1', wrk_list[1], iter_id)
#             writer.add_histogram('refine_scores1', refine_scores[1], iter_id)
#             true_labels = torch.where(yrk_list[1] == 1.0)
#             writer.add_histogram('trueX_1', refine_scores[1][true_labels], iter_id)
#             false_labels = torch.where(yrk_list[1] != 1.0)
#             writer.add_histogram('FalseX_1', refine_scores[1][false_labels], iter_id)
            for k in range(cfg.K):
#                 r_scores = torch.clamp(refine_scores[k], min=1e-6, max=1-1e-6)
                r_loss[k] = refineloss(refine_scores[k], 
                                       yrk_list[k],
                                       wrk_list[k])

            loss = b_loss + sum(r_loss)
            loss.backward()
            epoch_r_loss += sum(r_loss).item()

            iter_id += 1
#             if iter_id % cfg.TRAIN.ITER_SIZE == 0:
            optimizer.step()
            optimizer.zero_grad()
#         cls_ap = []
#         y_pred = np.array(y_pred)
#         y_true = np.array(y_true)
#         for i in range(20):
#             cls_ap.append(average_precision_score(y_true[:,i], y_pred[:,i])) 
#         record_info(f"Epoch {epoch} classify AP is {str(cls_ap)}", log_file)
#         record_info(f"Epoch {epoch} classify mAP is {str(sum(cls_ap)/20)}", log_file)

        record_info(f"Epoch {epoch} b_Loss is {epoch_b_loss/N}", log_file)
        record_info(f"Epoch {epoch} r_Loss is {epoch_r_loss/N}", log_file)
        if (epoch_b_loss + epoch_r_loss) / N  < eva_th:
            mAP = evaluate(model, test_loader, log_file)

        scheduler.step()
        if best_mAP < mAP:
            if best_mAP != 0: # the first model to save
                os.remove(cfg.PATH.PT_PATH + f"BestModel_{year}_{pretrained}_{best_epoch}.pt")
            best_model = model
            best_epoch = epoch
            best_mAP = mAP
            record_info(f'New Best Model: at Epoch {best_epoch}', log_file)
            torch.save({
               'whole_model_state_dict' : best_model.state_dict(),
               }, cfg.PATH.PT_PATH + f"BestModel_{year}_{pretrained}_{best_epoch}.pt")

        record_info("-" * 30, log_file)
        
        if epoch % save_step == 0:
            # disk space is not enough
            torch.save({'whole_model_state_dict' : model.state_dict(),}, 
                       cfg.PATH.PT_PATH + f"WholeModel_{year}_{pretrained}_{epoch}.pt")

    torch.save({
                'whole_model_state_dict' : model.state_dict(),
                }, cfg.PATH.PT_PATH + f"OK_WholeModel_{year}_{pretrained}_{epochs}_{comment}.pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
    
    