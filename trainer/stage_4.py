import torch
from tqdm import tqdm
from loguru import logger

from trainer.trainer import Trainer


class Trainer_4(Trainer):
    def __init__(self, args):
        self.stage = 4
        super(Trainer_4, self).__init__(args)
    
    def train_step(self, epoch, batch, num_steps):
        front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = batch
        #front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = front_sequence.to(self.device), leftrear_sequence.to(self.device), rightrear_sequence.to(self.device), nav_sequence.to(self.device), hist_trajectory_sequence.to(self.device), speed_limit_sequence.to(self.device), stop_sequence.to(self.device), traffic_convention_sequence.to(self.device), trajectory_sequence.to(self.device)
        bs = front_sequence.size(0)
        seq_length = front_sequence.size(1)
        
        hist_feature = torch.zeros((bs, 40, 128)).to(self.device)
        
        total_loss = 0
        for t in range(seq_length):
            num_steps += 1
            front, leftrear, rightrear, nav = front_sequence[:, t, :, :, :], leftrear_sequence[:, t, :, :, :], rightrear_sequence[:, t, :, :, :], nav_sequence[:, t, :, :, :]
            hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = hist_trajectory_sequence[:, t, :, :], speed_limit_sequence[:, t:t+1], stop_sequence[:, t, :], traffic_convention_sequence[:, t, :], trajectory_sequence[:, t, :, :]
            front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.to(self.device), leftrear.to(self.device), rightrear.to(self.device), nav.to(self.device), hist_trajectory.to(self.device), speed_limit.to(self.device), stop.to(self.device), traffic_convention.to(self.device), trajectory_label.to(self.device)
            front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.float(), leftrear.float(), rightrear.float(), nav.float(), hist_trajectory.float(), speed_limit.float(), stop.float(), traffic_convention.float(), trajectory_label.float()
            
            pred_cls, pred_trajectory, feature = self.model(front, leftrear, rightrear, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
            total_loss = self.compute_train_loss(epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, total_loss)
            feature = feature.clone().detach()
            hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
        return num_steps
    
    def validation_step(self, epoch, num_steps):
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = data
                #front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = front_sequence.to(self.device), leftrear_sequence.to(self.device), rightrear_sequence.to(self.device), nav_sequence.to(self.device), hist_trajectory_sequence.to(self.device), speed_limit_sequence.to(self.device), stop_sequence.to(self.device), traffic_convention_sequence.to(self.device), trajectory_sequence.to(self.device)

                bs = front_sequence.size(0)
                seq_length = front_sequence.size(1)
                
                hist_feature = torch.zeros((bs, 40, 128)).to(self.device)
                current_loss = 0
                for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                    front, leftrear, rightrear, nav = front_sequence[:, t, :, :], leftrear_sequence[:, t, :, :], rightrear_sequence[:, t, :, :], nav_sequence[:, t, :, :]
                    hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = hist_trajectory_sequence[:, t, :, :], speed_limit_sequence[:, t:t+1], stop_sequence[:, t, :], traffic_convention_sequence[:, t, :], trajectory_sequence[:, t, :, :]
                    front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.to(self.device), leftrear.to(self.device), rightrear.to(self.device), nav.to(self.device), hist_trajectory.to(self.device), speed_limit.to(self.device), stop.to(self.device), traffic_convention.to(self.device), trajectory_label.to(self.device)
                    front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.float(), leftrear.float(), rightrear.float(), nav.float(), hist_trajectory.float(), speed_limit.float(), stop.float(), traffic_convention.float(), trajectory_label.float()

                    pred_cls, pred_trajectory, feature = self.model(front, leftrear, rightrear, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
                    feature = feature.clone().detach()
                    hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
                    val_loss = self.compute_val_loss(epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, val_loss)
                logger.info("val | val_loss: {}", current_loss)
        self.model.train()
        