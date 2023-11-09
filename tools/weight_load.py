import time
import os
import sys
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from model import PlanModel


model = PlanModel(num_cls=5, num_pts=8).cuda()
state_dict = torch.load("D:\ETSPlan\epoch_6.pth")
print(state_dict.keys())
model.load_state_dict(state_dict, strict=True)
