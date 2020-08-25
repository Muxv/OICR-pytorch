import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.ops import roi_pool
from collections import OrderedDict 


def init_parameters(module):
    if type(module) in [nn.Linear]:
        torch.nn.init.normal_(module.weight, mean=0.0, std=1e-2)
        torch.nn.init.constant_(module.bias, 0)

def copy_parameters(src, target):
    assert src.weight.size() == target.weight.size()
    assert src.bias.size() == target.bias.size()
    src.weight = target.weight
    src.bias = target.bias
        
class Combined_VGG16(nn.Module):
    def __init__(self, K=3):
        super(Combined_VGG16, self).__init__()
        self.K = K
        vgg = torchvision.models.vgg16(pretrained=True)
        self.pretrained_features = nn.Sequential(*list(vgg.features._modules.values())[:23])
        self.new_features = nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu5_3', nn.ReLU(inplace=True)),
        ]))
        self.roi_size = (7, 7)
        self.roi_spatial_scale= 0.125
        copy_parameters(self.new_features.conv5_1, vgg.features[24])
        copy_parameters(self.new_features.conv5_2, vgg.features[26])
        copy_parameters(self.new_features.conv5_3, vgg.features[28])
        
        self.fc67 = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        self.c_softmax = nn.Softmax(dim=1)
        self.d_softmax = nn.Softmax(dim=0)
        
        self.ic_score1 = nn.Linear(4096, 21)
        self.ic_score2 = nn.Linear(4096, 21)
        self.ic_score3 = nn.Linear(4096, 21)
        
        self.ic_prob1 = nn.Softmax(dim=1)
        self.ic_prob2 = nn.Softmax(dim=1)
        self.ic_prob3 = nn.Softmax(dim=1)
        
            
    def forward(self, x, regions):
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        features = self.new_features(self.pretrained_features(x))
        pool_features = roi_pool(features, regions, self.roi_size, self.roi_spatial_scale).view(R, -1)
        fc7 = self.fc67(pool_features)
        c_score = self.c_softmax(self.fc8c(fc7))
        d_score = self.d_softmax(self.fc8d(fc7))
        proposal_scores = c_score * d_score
        
        ref_scores1 = self.ic_prob1(self.ic_score1(fc7))
        ref_scores2 = self.ic_prob2(self.ic_score2(fc7))
        ref_scores3 = self.ic_prob3(self.ic_score3(fc7))
        return ref_scores1, ref_scores2, ref_scores3, proposal_scores

    def init_model(self):
        self.fc8c.apply(init_parameters)
        self.fc8d.apply(init_parameters)
        self.ic_score1.apply(init_parameters)
        self.ic_score2.apply(init_parameters)
        self.ic_score3.apply(init_parameters)
        
        
