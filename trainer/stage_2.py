import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from trainer.trainer import Trainer


class Trainer_2(Trainer):
    def __init__(self, args):
        self.stage = 2
        super(Trainer_2, self).__init__(args)
        self.batch_size = args.batch_size[0]
        self.leftrear = torch.FloatTensor(np.zeros((self.batch_size, 3, 64, 64)), device=self.device)
        self.rightrear = torch.FloatTensor(np.zeros((self.batch_size, 3, 64, 64)), device=self.device)
        self.hist_feature = torch.FloatTensor(np.zeros((self.batch_size, 40, 128)), device=self.device)
        self.hist_trajectory = torch.FloatTensor(np.zeros((self.batch_size, 10, 2)), device=self.device)
        self.speed_limit = torch.FloatTensor(np.zeros((self.batch_size, 1)), device=self.device)
        self.stop = torch.FloatTensor(np.zeros((self.batch_size, 2)), device=self.device)
        self.traffic_convention = torch.FloatTensor(np.zeros((self.batch_size, 2)), device=self.device)
    
    def train_step(self, epoch, batch, num_steps):
        front_sequence, nav_sequence, trajectory_sequence = batch
        #front_sequence, nav_sequence, trajectory_sequence = front_sequence.to(self.device), nav_sequence.to(self.device), trajectory_sequence.to(self.device)
        seq_length = front_sequence.size(1)
        
        hist_feature = torch.zeros((self.batch_size, 40, 128)).to(self.device)
        
        total_loss = 0
        for t in range(seq_length):
            num_steps += 1
            front, nav, trajectory_label = front_sequence[:, t, :, :, :].to(self.device).float(), nav_sequence[:, t, :, :, :], trajectory_sequence[:, t, :, :].to(self.device).float()
            pred_cls, pred_trajectory, feature = self.model(front, self.leftrear, self.rightrear, nav, self.hist_feature, self.hist_trajectory, self.speed_limit, self.stop, self.traffic_convention)
            total_loss = self.compute_train_loss(epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, total_loss)
            feature = feature.clone().detach()
            hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
        return num_steps
    
    def validation_step(self, epoch, num_steps):
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                front_sequence, nav_sequence, trajectory_sequence = data
                #front_sequence, nav_sequence, trajectory_sequence = front_sequence.to(self.device), nav_sequence.to(self.device), trajectory_sequence.to(self.device)
                seq_length = front_sequence.size(1)
                
                hist_feature = torch.zeros((self.batch_size, 40, 128)).to(self.device)
                current_loss = 0
                for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                    front, nav, trajectory_label = front_sequence[:, t, :, :, :].to(self.device).float(), nav_sequence[:, t, :, :, :], trajectory_sequence[:, t, :, :].to(self.device).float()
                    pred_cls, pred_trajectory, feature = self.model(front, self.leftrear, self.rightrear, nav, self.hist_feature, self.hist_trajectory, self.speed_limit, self.stop, self.traffic_convention)
                    feature = feature.clone().detach()
                    hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
                    val_loss = self.compute_val_loss(epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, val_loss)
                logger.info("val | val_loss: {}", current_loss)
        self.model.train()
        