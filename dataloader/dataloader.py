from loguru import logger
import sys
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
import time
import torch

from dataloader.transform import world_to_frenet
    

class ETSMotion(Dataset):
    def __init__(self, dataset_path, stage):
        '''
        stage1: front;
        stage2: front, nav; 
        stage3: front, leftrear, rightrear, nav;
        stage4: full;
        '''
        self.stage = stage
        self.dataset_path = dataset_path
        self.segments_path = []
        self.intervals_path = os.listdir(self.dataset_path)
        self.intervals_num = len(self.intervals_path)
        logger.info("{} scenes in the dataset.".format(self.intervals_num))
        
        for interval_path in self.intervals_path:
            segments_path = os.listdir(os.path.join(self.dataset_path, interval_path))
            for segment_path in segments_path:
                self.segments_path.append(os.path.join(self.dataset_path, interval_path, segment_path))
        logger.info("{} segments in the dataset.".format(len(self.segments_path)))

    def __len__(self):
        return len(self.segments_path)
    
    def _load_stage_1(self, index):
        video_path = os.path.join(self.dataset_path, self.segments_path[index], 'video.npz')
        info_path = os.path.join(self.dataset_path, self.segments_path[index], 'info.npz')
        video = np.load(video_path)
        front = video['front'].transpose(0,3,1,2)
        info = np.load(info_path)['arr_0']
        
        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        trajectory_list = []
        video_len = len(info)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]
        
        for i in range(start_index, end_index):
            current_location = info[i, 0:2]
            current_rotation = info[i, 3]
            fut_trajectory = info[trajectory_index+i, 0:2]
            fut_trajectory = world_to_frenet(fut_trajectory, current_location, current_rotation)
            
            trajectory_list.append(fut_trajectory)
        
        front = front[start_index:end_index]
        front = torch.FloatTensor(front) / 255.0
        trajectory_list = torch.FloatTensor(trajectory_list)
        return front, trajectory_list
    
    def _load_stage_2(self, index):
        video_path = os.path.join(self.dataset_path, self.segments_path[index], 'video.npz')
        info_path = os.path.join(self.dataset_path, self.segments_path[index], 'info.npz')
        video = np.load(video_path)
        front = video['front'].transpose(0,3,1,2)
        nav = video['nav'].transpose(0,3,1,2)
        info = np.load(info_path)['arr_0']
        
        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        trajectory_list = []
        video_len = len(info)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]
        
        for i in range(start_index, end_index):
            current_location = info[i, 0:2]
            current_rotation = info[i, 3]
            fut_trajectory = info[trajectory_index+i, 0:2]
            fut_trajectory = world_to_frenet(fut_trajectory, current_location, current_rotation)
            
            trajectory_list.append(fut_trajectory)
        
        front = front[start_index:end_index]
        nav = nav[start_index:end_index]
        
        front = torch.FloatTensor(front) / 255.0
        nav = torch.FloatTensor(nav) / 255.0
        traffic_convention_list = torch.FloatTensor(traffic_convention_list)
        trajectory_list = torch.FloatTensor(trajectory_list)
        return front, nav, trajectory_list
    
    def _load_stage_3(self, index):
        video_path = os.path.join(self.dataset_path, self.segments_path[index], 'video.npz')
        info_path = os.path.join(self.dataset_path, self.segments_path[index], 'info.npz')
        video = np.load(video_path)
        front = video['front'].transpose(0,3,1,2)
        leftrear = video['leftrear'].transpose(0,3,1,2)
        rightrear = video['rightrear'].transpose(0,3,1,2)
        nav = video['nav'].transpose(0,3,1,2)
        info = np.load(info_path)['arr_0']
        
        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        trajectory_list = []
        video_len = len(info)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]
        
        for i in range(start_index, end_index):
            current_location = info[i, 0:2]
            current_rotation = info[i, 3]
            fut_trajectory = info[trajectory_index+i, 0:2]
            fut_trajectory = world_to_frenet(fut_trajectory, current_location, current_rotation)
            trajectory_list.append(fut_trajectory)
        
        front = front[start_index:end_index]
        leftrear = leftrear[start_index:end_index]
        rightrear = rightrear[start_index:end_index]
        nav = nav[start_index:end_index]
        
        front = torch.FloatTensor(front) / 255.0
        leftrear = torch.FloatTensor(leftrear) / 255.0
        rightrear = torch.FloatTensor(rightrear) / 255.0
        nav = torch.FloatTensor(nav) / 255.0
        trajectory_list = torch.FloatTensor(trajectory_list)
        return front, leftrear, rightrear, nav, trajectory_list
        
    def _load_stage_4(self, index):
        video_path = os.path.join(self.dataset_path, self.segments_path[index], 'video.npz')
        info_path = os.path.join(self.dataset_path, self.segments_path[index], 'info.npz')
        video = np.load(video_path)
        front = video['front'].transpose(0,3,1,2)
        leftrear = video['leftrear'].transpose(0,3,1,2)
        rightrear = video['rightrear'].transpose(0,3,1,2)
        nav = video['nav'].transpose(0,3,1,2)
        info = np.load(info_path)['arr_0']
        
        hist_trajectory_list, stop_list, traffic_convention_list = [], [], []
        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        trajectory_list = []
        video_len = len(info)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]
        
        for i in range(start_index, end_index):
            stop = np.eye(2)[int(info[i, 8])]
            traffic_convention = [0, 1]
            
            current_location = info[i, 0:2]
            current_rotation = info[i, 3]
            hist_trajectory = info[i-hist_trajectory_index, 0:2]
            fut_trajectory = info[trajectory_index+i, 0:2]
            hist_trajectory = world_to_frenet(hist_trajectory, current_location, current_rotation)
            fut_trajectory = world_to_frenet(fut_trajectory, current_location, current_rotation)
            
            stop_list.append(stop)
            traffic_convention_list.append(traffic_convention)
            hist_trajectory_list.append(hist_trajectory)
            trajectory_list.append(fut_trajectory)
        
        front = front[start_index:end_index]
        leftrear = leftrear[start_index:end_index]
        rightrear = rightrear[start_index:end_index]
        nav = nav[start_index:end_index]
        
        front = torch.HalfTensor(front) / 255.0
        leftrear = torch.HalfTensor(leftrear) / 255.0
        rightrear = torch.HalfTensor(rightrear) / 255.0
        nav = torch.HalfTensor(nav) / 255.0
        speed_limit_list = torch.FloatTensor(info[:, 7]) / 25
        stop_list = torch.FloatTensor(stop_list)
        traffic_convention_list = torch.FloatTensor(traffic_convention_list)
        hist_trajectory_list = torch.FloatTensor(hist_trajectory_list)
        trajectory_list = torch.FloatTensor(trajectory_list)
        return front, leftrear, rightrear, nav, hist_trajectory_list, speed_limit_list, stop_list, traffic_convention_list, trajectory_list
    
    def __getitem__(self, index):
        if self.stage == 1:
            front, trajectory_list = self._load_stage_1(index)
            return front, trajectory_list
        elif self.stage == 2:
            front, nav, trajectory_list = self._load_stage_2(index)
            return front, nav, trajectory_list
        elif self.stage == 3:
            front, leftrear, rightrear, nav, trajectory_list = self._load_stage_3(index)
            return front, leftrear, rightrear, nav, trajectory_list
        elif self.stage == 4:
            front, leftrear, rightrear, nav, hist_trajectory_list, speed_limit_list, stop_list, traffic_convention_list, trajectory_list = self._load_stage_4(index)
            return front, leftrear, rightrear, nav, hist_trajectory_list, speed_limit_list, stop_list, traffic_convention_list, trajectory_list
        
    
if __name__ == '__main__':
    etsmotion = ETSMotion("D:\ETSMotion\ETSMotion", stage=4)
    train_loader = DataLoader(dataset=etsmotion, batch_size=3, shuffle=False, num_workers=1)
    t0 = time.time()
    for data in train_loader:
        front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = data
    t1 = time.time()
    logger.info("Total take {}s to load dataset", t1-t0)
    front_sequence = front_sequence.numpy()[0].transpose(0,3,1,2)
    hist_trajectory_sequence = hist_trajectory_sequence.numpy()[0]
    trajectory_sequence = trajectory_sequence.numpy()[0]
    print(front_sequence.shape)
    print(trajectory_sequence.shape)
    print(hist_trajectory_sequence.shape)
    for i in range(len(trajectory_sequence)):
        front = front_sequence[i]
        hist_trajectory = hist_trajectory_sequence[i]
        trajectory = trajectory_sequence[i]
        trajectory_total = np.concatenate((hist_trajectory, np.zeros((1, 2)), trajectory), axis=0) * 4
        img = np.zeros((600, 200, 3), np.uint8)
        cv2.polylines(img, np.int32([trajectory_total + np.array([100, 400])]), False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (100, 400), 3, (0, 0, 255), -1)
        cv2.imshow('front', np.uint8(front*255))
        cv2.imshow('trajectory', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        