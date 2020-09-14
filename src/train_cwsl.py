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
from oicr_layer import *



import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.benchmark = True

# gradient_writer = SummaryWriter()

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train OICR")
    parse.add_argument(
        "--year", type=str, default='2007', help="Use which year of VOC"
    )
    parse.add_argument(
        "--pretrained", type=str, default='vgg16', help="which pretrained model to use"
    )
    parse.add_argument(
        "--comment",  type=str, default='', help="add comment"
    )

    setup_seed(cfg.SEED)

    save_step = 5
    args = parse.parse_args()
    year = args.year
    pretrained = args.pretrained
    comment = args.comment
    oicr = None
    midn = None
    
    if pretrained == 'alexnet':
        model = Combined_Alexnet(cfg.K)
    if pretrained == 'vgg16':
        model = Combined_VGG16(cfg.K)
    
    eva_th = 10

    
#     lr = cfg.TRAIN.LR
    lr = 1e-4
    lr_step = 100
#     epochs = cfg.TRAIN.EPOCH
    epochs = 50
    start_epoch = 31
    
    
    log_file = cfg.PATH.LOG_PATH + f"Model_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M')+ ".txt"
    record_info(f"Full Epoch {epochs}", log_file)
    record_info(f"Base LR {lr}", log_file)
    record_info("-" * 30, log_file)

    model.to(cfg.DEVICE)
    model.init_model()
    
    checkpoints = torch.load(cfg.PATH.PT_PATH + "cwsl_WholeModel_2007_vgg16_30.pt")
    model.load_state_dict(checkpoints['whole_model_state_dict'])

    
    trainval = VOCDectectionDataset("~/data/", year, 'trainval')
    train_loader = data.DataLoader(trainval, 
                                   cfg.TRAIN.BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)
    testdata = VOCDectectionDataset("~/data/", year, 'test', small_box=False)
    test_loader = data.DataLoader(testdata,
                                  1,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)

#     bias_params = []
#     bias_param_names = []
#     nonbias_params = []
#     nonbias_param_names = []
#     nograd_param_names = []
#     for key, value in model.named_parameters():
#         if value.requires_grad:
#             if 'bias' in key:
#                 bias_params.append(value)
#                 bias_param_names.append(key)
#             else:
#                 nonbias_params.append(value)
#                 nonbias_param_names.append(key)
                
#     params = [
#         {'params': nonbias_params,
#          'lr': lr,
#          'weight_decay': cfg.TRAIN.WD},
#         {'params': bias_params,
#          'lr': lr * (cfg.TRAIN.BIAS_DOUBLE_LR + 1),
#          'weight_decay':  0},
#     ]
    
#     optimizer = optim.SGD(params,
#                           momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.TRAIN.WD)
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
    
#     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    
    for epoch in tqdm(range(start_epoch, epochs+1), "Total"):
        epoch_b_loss = 0.0
        epoch_r_loss = 0.0
        for img, gt_box, gt_count, regions in tqdm(train_loader, f"Epoch {epoch}"):
            img = img.to(cfg.DEVICE)  # 1, 3, h ,w 
            regions = regions.to(cfg.DEVICE) # 1, R, 4
            R = regions.size()[1] # R
            gt_count = gt_count.to(cfg.DEVICE) # 1, C
            gt_label = (gt_count > 0.).float()
            
            ref_scores1, ref_scores2, ref_scores3, proposal_scores = model(img, regions)
            cls_scores = torch.sum(proposal_scores, dim=0)
            cls_scores = torch.clamp(cls_scores, min=0, max=1)
            
            b_loss = bceloss(cls_scores, gt_label[0])
            epoch_b_loss += b_loss.item()


            
            xr0 = torch.zeros((R, 21)).to(cfg.DEVICE) # xj0
            xr0[:, :20] = proposal_scores.clone()
            xrk_list = []
            xrk_list.append(xr0)
            xrk_list.append(ref_scores1.clone())
            xrk_list.append(ref_scores2.clone())
            

#             wrk_list, yrk_list = oicr_algorithm(xrk_list, gt_label, regions, cfg.K)
            wrk_list, yrk_list = oicr_crs_algorithm(xrk_list, gt_count, regions, cfg.K, cfg.T, cfg.MAXK)
    
            r_loss_1 = refineloss(ref_scores1, 
                                  yrk_list[0],
                                  wrk_list[0])
            r_loss_2 = refineloss(ref_scores2, 
                                  yrk_list[1],
                                  wrk_list[1])
            r_loss_3 = refineloss(ref_scores3, 
                                  yrk_list[2],
                                  wrk_list[2])

            loss = b_loss + r_loss_1 + r_loss_2 + r_loss_3
            
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()
            
            loss.backward()
            epoch_r_loss += (r_loss_1 + r_loss_2 + r_loss_3).item()

            iter_id += 1
            if iter_id % cfg.TRAIN.ITER_SIZE == 0:
                optimizer.step()
                for param in model.parameters():
                    param.grad = None
                
                iter_id = 0

        record_info(f"Epoch {epoch} b_Loss is {epoch_b_loss/N}", log_file)
        record_info(f"Epoch {epoch} r_Loss is {epoch_r_loss/N}", log_file)
        
        scheduler.step()
       
        if epoch % save_step == 0:
            # disk space is not enough
            record_info(f'Model Saved: at Epoch {epoch}', log_file)
            
            if os.path.exists(cfg.PATH.PT_PATH + f"{comment}_WholeModel_{year}_{pretrained}_{epoch-save_step}.pt"):
                os.remove(cfg.PATH.PT_PATH + f"{comment}_WholeModel_{year}_{pretrained}_{epoch-save_step}.pt")
            
            torch.save({'whole_model_state_dict' : model.state_dict(),}, 
                       cfg.PATH.PT_PATH + f"{comment}_WholeModel_{year}_{pretrained}_{epoch}.pt")
        
        if (epoch_b_loss + epoch_r_loss) / N  < eva_th:
            mAP = evaluate(model, test_loader, log_file)


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
        

    torch.save({
                'whole_model_state_dict' : model.state_dict(),
                }, cfg.PATH.PT_PATH + f"OK_WholeModel_{year}_{pretrained}_{epochs}_{comment}.pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
    
    
