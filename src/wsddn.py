import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from utils import *
from torchvision.ops import roi_pool

class WSDDN_Alexnet(nn.Module):
    def __init__(self):
        super(WSDDN_Alexnet, self).__init__()        
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.features = nn.Sequential(*list(alexnet.features._modules.values())[:-1])
        self.fc67 = nn.Sequential(*list(alexnet.classifier._modules.values())[:-1])
        
        self.roi_output_size = (6, 6)

        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        self.cls_softmax = nn.Softmax(dim=1)
        self.det_softmax = nn.Softmax(dim=0)
        
    def forward(self, x, regions, scores=None):
        #   x    : bs, c ,h, w
        # regions: bs, R, 4
        #  scores: bs, R
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        features = self.features(x) # bs, 256， h/16, w/16
        pool_features = roi_pool(features, regions, self.roi_output_size, 1.0/16).view(R, -1) # R, 256, 6, 6
        
        if scores is not None:
            pool_features = pool_features * (10 * scores[0] + 1)

        fc7 = self.fc67(pool_features)
        # fc8x(out)   R, 20
        cls_score = self.cls_softmax(self.fc8c(fc7))
        det_score = self.det_softmax(self.fc8d(fc7))
        combined = cls_score * det_score

        return combined, fc7
    def spatial_regulariser(self, regions, fc7, combine_scores, labels):
        iou_th = 0.6
        K = 10 #  top 10 scores
        reg = 0
        cls_num = 0
        for c in range(20):
            # extract positive ones
            if labels[c].item() == 0:
                continue
            cls_num += 1
            topk_scores, topk_filter = combine_scores[:, c].topk(K, dim=0)
            topk_boxes = regions[topk_filter]
            topk_fc7 = fc7[topk_filter]

            # get box with the best box | iou > 0.6
            iou_mask = one2allbox_iou(topk_boxes[0:1, :], topk_boxes).view(K)
            iou_mask = (iou_mask > iou_th).float()

            fc7_diff = topk_fc7 - topk_fc7[0]
            score_diff = topk_scores.detach().view(K, 1)

            diff = fc7_diff * score_diff

            reg += 0.5 * (torch.pow(diff, 2).sum(1) * iou_mask).sum()            
        return reg/cls_num

    
class WSDDN_VGG16(nn.Module):
    def __init__(self):
        super(WSDDN_VGG16, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.fc67 = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        self.roi_output_size = (7, 7)
        
                
        self.cls_softmax = nn.Softmax(dim=1)
        self.det_softmax = nn.Softmax(dim=0)
        
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        
    def forward(self, x, regions, scores=None):
        #   x    : bs, c ,h, w
        # regions: bs, R, 4
        #  scores: bs, R
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        features = self.features(x) # bs, 256， h/16, w/16
        pool_features = roi_pool(features, regions, self.roi_output_size, 1.0/16).view(R, -1) # R, 256, 6, 6
        
        if scores is not None:
            pool_features = pool_features * (10 * scores[0] + 1)

        fc7 = self.fc67(pool_features)
        # fc8x(out)   R, 20
        cls_score = self.cls_softmax(self.fc8c(fc7))
        det_score = self.det_softmax(self.fc8d(fc7))
        combined = cls_score * det_score
        return combined, fc7

    def spatial_regulariser(self, regions, fc7, combine_scores, labels):
        iou_th = 0.6
        K = 10 #  top 10 scores
        reg = 0
        cls_num = 0
        for c in range(20):
            # extract positive ones
            if labels[c].item() == 0:
                continue
            cls_num += 1
            topk_scores, topk_filter = combine_scores[:, c].topk(K, dim=0)
            topk_boxes = regions[topk_filter]
            topk_fc7 = fc7[topk_filter]

            # get box with the best box | iou > 0.6
            iou_mask = one2allbox_iou(topk_boxes[0:1, :], topk_boxes).view(K)
            iou_mask = (iou_mask > iou_th).float()

            fc7_diff = topk_fc7 - topk_fc7[0]
            score_diff = topk_scores.detach().view(K, 1)

            diff = fc7_diff * score_diff

            reg += 0.5 * (torch.pow(diff, 2).sum(1) * iou_mask).sum()            
        return reg/cls_num