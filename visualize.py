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
import random
import onnxruntime

from dataloader.transform import world_to_frenet


class ETSPlan(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.input0_name = self.session.get_inputs()[0].name
        self.input1_name = self.session.get_inputs()[1].name
        self.input2_name = self.session.get_inputs()[2].name
        self.input3_name = self.session.get_inputs()[3].name
        self.input4_name = self.session.get_inputs()[4].name
        self.input5_name = self.session.get_inputs()[5].name
        self.input6_name = self.session.get_inputs()[6].name
        self.input7_name = self.session.get_inputs()[7].name
        self.input8_name = self.session.get_inputs()[8].name
        self.output0_name = self.session.get_outputs()[0].name
        self.output1_name = self.session.get_outputs()[1].name
        self.output2_name = self.session.get_outputs()[2].name
        self.hist_feature = np.zeros((1, 40, 128), dtype=np.float32)
        
    def infer(self, front, leftrear, rightrear, nav, 
              hist_trajectory, speed_limit, stop_list, traffic_convention):
        pred_cls, pred_trajectory, feature = self.session.run([self.output0_name,
                                                          self.output1_name,
                                                          self.output2_name,],
                                                         {self.input0_name: front,
                                                          self.input1_name: leftrear,
                                                          self.input2_name: rightrear,
                                                          self.input3_name: nav,
                                                          self.input4_name: self.hist_feature,
                                                          self.input5_name: hist_trajectory,
                                                          self.input6_name: speed_limit,
                                                          self.input7_name: stop_list,
                                                          self.input8_name: traffic_convention,
                                                         })
        trajectory = pred_trajectory[0, np.argmax(pred_cls[0]), :, :]
        abandon_trajectorys = pred_trajectory[0]
        self.hist_feature = np.concatenate((self.hist_feature, feature), axis=1)[:, 1:, :]
        #t2 = time.time()
        #print('infer: ', t2-t1)
        return trajectory, abandon_trajectorys
    
    
def visualize(trajectory, hist_trajectory, trajectory_pred, abandon_trajectorys):
    K = 5
    img = np.zeros((600, 200, 3), np.uint8)
    trajectory_true = np.concatenate((hist_trajectory, np.zeros((1, 2)), trajectory), axis=0) * K
    trajectory_pred = trajectory_pred * K
    cv2.polylines(img, np.int32([trajectory_true + np.array([100, 400])]), False, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.polylines(img, np.int32([trajectory_pred + np.array([100, 400])]), False, (0, 255, 0), 1, cv2.LINE_AA)
    #for abandon_trajectory in abandon_trajectorys:
    #    cv2.polylines(img, np.int32([abandon_trajectory + np.array([100, 400])]), False, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.circle(img, (100, 400), 3, (0, 0, 255), -1)
    return img


def main(dataset_path, onnx_path):
    etsplan = ETSPlan(onnx_path)
    intervals_path = os.listdir(dataset_path)
    
    choice_intervals = random.choice(intervals_path)
    interval_path = os.listdir(os.path.join(dataset_path, choice_intervals))
    for segment_path in interval_path:
        video_path = os.path.join(dataset_path, choice_intervals, segment_path, 'video.npz')
        info_path = os.path.join(dataset_path, choice_intervals, segment_path, 'info.npz')
        video = np.load(video_path)
        front_sequence = video['front'].transpose(0,3,1,2)
        leftrear_sequence = video['leftrear'].transpose(0,3,1,2)
        rightrear_sequence = video['rightrear'].transpose(0,3,1,2)
        nav_sequence = video['nav'].transpose(0,3,1,2)
        info_sequence = np.load(info_path)['arr_0']

        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        video_len = len(info_sequence)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]

        for i in range(start_index, end_index):
            front = front_sequence[i:i+1,:,:,:] / 255.0
            leftrear = leftrear_sequence[i:i+1,:,:,:] / 255.0
            rightrear = rightrear_sequence[i:i+1,:,:,:] / 255.0
            nav = nav_sequence[i:i+1,:,:,:] / 255.0
            speed_limit = np.expand_dims(info_sequence[i:i+1, 7] / 25, axis=0)
            stop = np.array([np.eye(2)[int(info_sequence[i, 8])]])
            traffic_convention = np.array([[0, 1]])
            
            current_location = info_sequence[i, 0:2]
            current_rotation = info_sequence[i, 3]
            hist_trajectory = info_sequence[i-hist_trajectory_index, 0:2]
            fut_trajectory = info_sequence[trajectory_index+i, 0:2]
            hist_trajectory = np.expand_dims(world_to_frenet(hist_trajectory, current_location, current_rotation), axis=0)
            fut_trajectory = np.expand_dims(world_to_frenet(fut_trajectory, current_location, current_rotation), axis=0)
            
            front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention = np.array(front, dtype=np.float32), np.array(leftrear, dtype=np.float32), np.array(rightrear, dtype=np.float32), np.array(nav, dtype=np.float32), np.array(hist_trajectory, dtype=np.float32), np.array(speed_limit, dtype=np.float32), np.array(stop, dtype=np.float32), np.array(traffic_convention, dtype=np.float32)
            trajectory_pred, abandon_trajectory = etsplan.infer(front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention)
            
            img = visualize(fut_trajectory[0], hist_trajectory[0], trajectory_pred, abandon_trajectory)
            cv2.imshow('trajectory', img)
            cv2.imshow('front', np.uint8(front[0]*255).transpose(1,2,0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

if __name__ == '__main__':
    dataset_path = "D:\ETSMotion\ETSMotion"
    onnx_path = "D:\ETSPlan\etsplan.onnx"
    main(dataset_path, onnx_path)