import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.ops import roi_pool
from collections import OrderedDict 

def init_parameters(module):
    if type(module) in [nn.Conv2d, nn.Linear]:
        torch.nn.init.normal_(module.weight, mean=0.0, std=1e-2)
        torch.nn.init.zeros_(module.bias)
        
class OICR_Alexnet(nn.Module):
    def __init__(self, K=3):
        super(OICR_Alexnet, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.K = K
        self.pretrained_features = nn.Sequential(*list(alexnet.features._modules.values())[:5])
        self.new_features = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)),
            ('relu5', nn.ReLU(inplace=True)),
        ]))
        self.roi_size = (6, 6)
        self.roi_spatial_scale= 0.125
        self.fc67 = nn.Sequential(*list(alexnet.classifier._modules.values())[:-1])
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        self.c_softmax = nn.Softmax(dim=1)
        self.d_softmax = nn.Softmax(dim=0)
        for i in range(self.K):
            self.add_module(
                f'refine{i}',
                nn.Sequential(OrderedDict([
                    (f'ic_score{i}', nn.Linear(4096, 21)),
                    (f'ic_probs{i}', nn.Softmax(dim=1))
                ])))
            
    def forward(self, x, regions):
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        features = self.new_features(self.pretrained_features(x))
        pool_features = roi_pool(features, regions, self.roi_size, self.roi_spatial_scale).view(R, -1)
        fc7 = self.fc67(pool_features)
        c_score = self.fc8c(self.c_softmax(fc7))
        d_score = self.fc8d(self.d_softmax(fc7))
        proposal_scores = c_score * d_score
        proposal_scores = torch.clamp(proposal_scores, min=0.0, max=1.0)
        refine_scores = []
        for i in range(self.K):
            refine_scores.append(self._modules[f'refine{i}'](fc7))
        return proposal_scores, refine_scores
    
    def init_model(self):
        self.new_features.apply(init_parameters)
        self.fc8c.apply(init_parameters)
        self.fc8d.apply(init_parameters)
        K = self.K
        for i in range(K):
            self._modules[f'refine{i}'].apply(init_parameters)

class OICR_VGG16(nn.Module):
    def __init__(self, K=3):
        super(OICR_VGG16, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.K = K
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
        self.fc67 = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        self.c_softmax = nn.Softmax(dim=1)
        self.d_softmax = nn.Softmax(dim=0)
        for i in range(self.K):
            self.add_module(
                f'refine{i}',
                nn.Sequential(OrderedDict([
                    (f'ic_score{i}', nn.Linear(4096, 21)),
                    (f'ic_probs{i}', nn.Softmax(dim=1))
                ])))
            
    def forward(self, x, regions):
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        features = self.new_features(self.pretrained_features(x))
        pool_features = roi_pool(features, regions, self.roi_size, self.roi_spatial_scale).view(R, -1)
        fc7 = self.fc67(pool_features)
        c_score = self.fc8c(self.c_softmax(fc7))
        d_score = self.fc8d(self.d_softmax(fc7))
        proposal_scores = c_score * d_score
        proposal_scores = torch.clamp(proposal_scores, min=0.0, max=1.0)
        refine_scores = []
        for i in range(self.K):
            refine_scores.append(self._modules[f'refine{i}'](fc7))
        return proposal_scores, refine_scores

    def init_model(self):
        self.new_features.apply(init_parameters)
        self.fc8c.apply(init_parameters)
        self.fc8d.apply(init_parameters)
        K = self.K
        for i in range(K):
            self._modules[f'refine{i}'].apply(init_parameters)