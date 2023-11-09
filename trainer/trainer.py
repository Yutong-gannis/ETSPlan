import torch
from loguru import logger
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataloader.dataloader import ETSMotion
from model.etsplan import ETSPlan
from loss import MultipleTrajectoryPredictionLoss


class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.split = 0.8
        self.writer = SummaryWriter()
        self.device = torch.device(self.args.device)
        self.model = ETSPlan(num_cls=self.args.num_cls, num_pts=self.args.num_pts).to(self.device)
        self.loss = MultipleTrajectoryPredictionLoss(self.args.mtp_alpha, self.args.num_cls, self.args.num_pts, distance_type='angle')
        if self.args.resume:
            if self.args.load_encoder:
                self._load_encoder_weight()
            else:
                self._load_weight()
        self._get_dataloader()
        self._configure_optimizers()
    
    def _load_weight(self):
        state_dict = torch.load(self.args.resume)
        self.model.load_state_dict(state_dict, strict=True)
    
    def _load_encoder_weight(self):
        rl_state_dict = torch.load(self.args.resume, map_location='cpu')
        self._load_state_dict(self.model.imgencoder, rl_state_dict, 'imgencoder')
        self._load_state_dict(self.model.leftrearencoder, rl_state_dict, 'leftrearencoder')
        self._load_state_dict(self.model.rightrearencoder, rl_state_dict, 'rightrearencoder')
        self._load_state_dict(self.model.navencoder, rl_state_dict, 'navencoder')
    
    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)
        
    def _configure_optimizers(self):
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.01)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, )
        else:
            raise NotImplementedError
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.9)
        
    def _get_dataloader(self):
        etsmotion = ETSMotion(self.args.dataset, self.stage)
        train_size = int(len(etsmotion) * self.split)
        val_size = len(etsmotion) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(etsmotion, [train_size, val_size])
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size[self.stage-1], shuffle=True, num_workers=self.args.num_workers[self.stage-1])
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size[self.stage-1], shuffle=True, num_workers=self.args.num_workers[self.stage-1])
    
    def save_checkpoint(self, epoch):
        ckpt_path = 'stage_{}-epoch_{}.pth'.format(self.stage, epoch)
        torch.save(self.model.state_dict(), ckpt_path)
        print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))
        
    def forward(self, batch):
        pass
    
    def compute_train_loss(self, epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, total_loss):
        cls_loss, reg_loss = self.loss(pred_cls, pred_trajectory, trajectory_label)
        total_loss += (cls_loss + self.args.mtp_alpha * reg_loss.mean()) / self.args.optimize_per_n_step
    
        if num_steps % self.args.log_per_n_step == 0:
            self.writer.add_scalar('train/epoch', epoch, num_steps)
            self.writer.add_scalar('loss/cls', cls_loss, num_steps)
            self.writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
            self.writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
            self.writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
            self.writer.add_scalar('param/lr', self.optimizer.param_groups[0]['lr'], num_steps)
            
            logger.info("step: {} | total_loss: {} | cls_loss: {} | reg_loss: {} | reg_x_loss: {} | reg_y_loss: {} | lr: {}", 
                        num_steps, total_loss, cls_loss, reg_loss.mean(), reg_loss[0], reg_loss[1], self.optimizer.param_groups[0]['lr'])

        if num_steps % self.args.optimize_per_n_step == 0:
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            total_loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('loss/total', total_loss, num_steps)
            total_loss = 0
        return total_loss
    
    def compute_val_loss(self, epoch, num_steps, pred_cls, pred_trajectory, trajectory_label, val_loss):
        cls_loss, reg_loss = self.loss(pred_cls, pred_trajectory, trajectory_label)
        val_loss += (cls_loss + self.args.mtp_alpha * reg_loss.mean()) / self.args.optimize_per_n_step
    
        self.writer.add_scalar('val/step', epoch, num_steps)
        self.writer.add_scalar('val_loss/step', val_loss, num_steps)
        return val_loss
    
    def run(self):
        num_steps = 0
        for epoch in range(self.args.epochs[self.stage-1]):
            logger.info("Epoch: {}", epoch)
            for data in self.train_loader:
                num_steps = self.train_step(epoch, data, num_steps)
            self.lr_scheduler.step()
            if (epoch + 1) % self.args.val_per_n_epoch == 0:
                self.save_checkpoint(epoch)
                self.validation_step(epoch, num_steps)