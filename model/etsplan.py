import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import ImgEncoder, RearEncoder, NAVEncoder, HistEncoder, ActionEncoder
from model.block import ResBlock


class ETSPlan(nn.Module):
    def __init__(self, num_cls=5, num_pts=8):
        super().__init__()
        self.num_cls = num_cls
        self.num_pts = num_pts
        self.imgencoder = ImgEncoder()
        self.leftrearencoder = RearEncoder()
        self.rightrearencoder = RearEncoder()
        self.navencoder = NAVEncoder()
        self.histencoder = HistEncoder()
        self.actionencoder = ActionEncoder()
        self.backbone = Backbone()
        self.neck = Neck()
        self.fc = nn.Sequential(nn.Linear(640, 128),
                                nn.ReLU())

    def forward(self, img, left_rear_img, right_rear_img, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention):
        img_feature = self.imgencoder(img)  # batch x 512
        left_rear_feature = self.leftrearencoder(left_rear_img)  # batch x 64
        right_rear_feature = self.rightrearencoder(right_rear_img)  # batch x 64
        frame_feature = torch.cat((img_feature, left_rear_feature, right_rear_feature), dim=1) # batch x 640
        
        hist_feature = self.histencoder(hist_feature)  # batch x 1024
        nav_feature = self.navencoder(nav, speed_limit, stop, traffic_convention)  # batch x 256
        hist_trajectory_feature = self.actionencoder(hist_trajectory)  # batch x 128
        feature = torch.cat((frame_feature, hist_feature, nav_feature, hist_trajectory_feature), dim=1)  # batch x 2048
        
        feature = self.backbone(feature) # batch x 256
        #action = self.neck(feature)  # batch x 2
        pred = self.neck(feature)  # batch x (5 x 8 x 2 + 5)
        pred_cls = pred[:, :self.num_cls]
        pred_trajectory = pred[:, self.num_cls:].reshape(-1, self.num_cls, self.num_pts, 2)

        #pred_xs = pred_trajectory[:, :, :, 0:1].exp()
        #pred_ys = pred_trajectory[:, :, :, 1:2].sinh()
        #pred_zs = pred_trajectory[:, :, :, 2:3]
        #pred_trajectory, torch.cat((pred_xs, pred_ys, pred_zs), dim=3)
        pred_xs = pred_trajectory[:, :, :, 0:1].sinh()
        pred_ys = pred_trajectory[:, :, :, 1:2].square()
        pred_trajectory, torch.cat((pred_xs, pred_ys), dim=3)
        
        buffer = self.fc(frame_feature).reshape((-1, 1, 128))
        # print("model output:" + str(action.shape))
        return pred_cls, pred_trajectory, buffer


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU())
        self.resblock_1 = ResBlock(512, 1024, 512)
        self.resblock_2 = ResBlock(512, 1024, 512)
        self.resblock_3 = ResBlock(512, 1024, 512)
        
    def forward(self, feature):
        feature = self.fc(feature)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        feature = self.resblock_3(feature)
        return feature
    
    
class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 1024),
                                   nn.ReLU())
        self.fc2 = nn.Linear(2048, 5*(8*2+1))
        self.resblock = ResBlock(1024, 1024, 2048)
        
    def forward(self, x):
        feature = self.fc1(x)
        feature = self.resblock(feature)
        feature = self.fc2(feature)
        return feature
    

if __name__ == '__main__':
    model = ETSPlan().cuda().half()
    
    batch = 5
    img = torch.zeros((batch, 3, 128, 512)).cuda().half()
    left_rear_img = torch.zeros((batch, 3, 64, 64)).cuda().half()
    right_rear_img = torch.zeros((batch, 3, 64, 64)).cuda().half()
    nav = torch.zeros((batch, 3, 64, 64)).cuda().half()
    hist_feature = torch.zeros((batch, 49, 128)).cuda().half()
    hist_trajectory = torch.zeros((batch, 10, 2)).cuda().half()
    speed_limit = torch.zeros((batch, 1)).cuda().half()
    stop = torch.zeros((batch, 2)).cuda().half()
    traffic_convention = torch.zeros((batch, 2)).cuda().half()
    cls, trajectory, feature = model(img, left_rear_img, right_rear_img, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
    print("pred shape:" + str(trajectory.shape))