
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from efficientnet_pytorch import EfficientNet
from model.block import ResBlock


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        input_dim = 1408
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b1', in_channels=3)
        #input_dim = 1280
        #self.backbone = mobilenet_v3_large(pretrained=True)
        #input_dim = 960
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 32, 1),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.ELU())
        self.resblock = ResBlock(2048, 512, 512)
        
    def forward(self, x):
        feature = self.backbone.extract_features(x)
        #feature = self.backbone.features(x)
        feature = self.conv(feature)
        feature = self.resblock(feature)
        # print("backbone output:" + str(feature.shape))
        return feature  # batch x 512
    

class RearEncoder(nn.Module):
    def __init__(self):
        super(RearEncoder, self).__init__()
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        #input_dim = 1408
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
        #input_dim = 1280
        #self.backbone = mobilenet_v3_large(pretrained=True)
        #input_dim = 960
        self.backbone = mobilenet_v3_small(pretrained=True)
        input_dim = 576
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 32, 1),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.ELU())
        self.resblock = ResBlock(32*4, 512, 64)
        
    def forward(self, x):
        #feature = self.backbone.extract_features(x)
        feature = self.backbone.features(x)
        #print(feature.shape)
        feature = self.conv(feature)
        #print(feature.shape)
        feature = self.resblock(feature)
        # print("backbone output:" + str(feature.shape))
        return feature  # batch x 64


class NAVEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        #input_dim = 1408
        #self.backbone = mobilenet_v3_large(pretrained=True)
        #input_dim = 960
        self.backbone = mobilenet_v3_small(pretrained=True)
        input_dim = 576
        self.nav_head = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 24, 1),
            nn.BatchNorm2d(24),
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.ELU(),
        )
        self.resblock_1 = ResBlock(256, 512, 256)
        self.resblock_2 = ResBlock(256, 512, 256)
        self.fc1 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256+1+2+2, 256),
                                 nn.ReLU())
        
    def forward(self, nav, speed_limit, stop, traffic_convention):
        #feature = self.backbone.extract_features(x)
        feature = self.backbone.features(nav)
        feature = self.nav_head(feature)
        #print(feature.shape)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        feature = self.fc1(feature)
        feature = torch.cat((feature, speed_limit, stop, traffic_convention), dim=1)
        out = self.fc2(feature)
        return out
    

class ActionEncoder(nn.Module):
    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.resblock_1 = ResBlock(20, 512, 512)
        self.resblock_2 = ResBlock(512, 512, 128)
        
    def forward(self, action):
        feature = self.flatten(action)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        return feature


class HistEncoder(nn.Module):
    def __init__(self):
        super(HistEncoder, self).__init__()
        self.index = [-5, -10, -15, -20, -25, -30, -35, -40]
        self.flatten = nn.Flatten()
        self.resblock = ResBlock(1024, 1024, 1024)
        self.fc = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        
    def forward(self, hist_feature):
        feature = hist_feature[:, self.index, :]
        feature = self.flatten(feature)
        feature = self.resblock(feature)
        feature = self.relu(self.fc(feature))
        return feature