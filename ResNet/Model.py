import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from libauc.losses import APLoss, AUCMLoss
from libauc.optimizers import SOAP, PESG

from .NetWork import ResNet, resnet18
from .ImageUtils import parse_record, radio_process

""" This script defines the training, validation and testing process.
"""

@dataclass
class Config:
    batch_size: int = 128
    num_classes: int = 10
    save_interval: int = 1
    weight_decay: float = 2e-4
    modeldir: str = 'checkpoints'
    loss: str = 'ce'
    optim: str = 'adam'
    lr: float = 1e-3
    dataset: float = 'cifar'
    gamma: float = 0.8
    img_dim: int = 32
    channels: int = 3
    dataset_size: int = 50_000
    scheduler: str = 'default'
    device: str = 'mps'
    max_epochs: int = 40
    res_on: bool = True
    norm_on: bool = True
    config_id: int = 0
    seed: int = 42


class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        num_classes = 1 if config.loss in ('ap', 'aucm') else config.num_classes
        self.network = resnet18(
            norm_on=config.norm_on, 
            res_on=config.res_on, 
            num_classes=num_classes, 
            channels=config.channels,
            last_activation = 'sigmoid' if config.loss in ('ap', 'aucm') else None
        ).to(config.device)
        self.device = torch.device(config.device)

        # set loss
        if config.loss == 'ap':
            self.loss = APLoss(config.dataset_size, gamma=config.gamma, margin=0.6, device=self.device)
        elif config.loss == 'aucm':
            self.loss = AUCMLoss(margin=1.0, imratio=None, version='v1', device=self.device)
        else:
            self.loss = nn.CrossEntropyLoss()

        # set optimizer
        if config.optim == 'soap':
            self.optimizer = SOAP(self.network.parameters(), lr=config.lr, mode='adam', weight_decay=config.weight_decay, device=self.device)
        elif config.optim == 'pesg':
            self.optimizer = PESG(self.network.parameters(), self.loss, config.lr, mode='adam', weight_decay=config.weight_decay, epoch_decay=1e-3, device=self.device)
        else:
            self.optimizer = optim.Adam(self.network.parameters(),  \
                             self.config.lr, weight_decay=self.config.weight_decay)

        # set scheduler
        if config.optim == 'pesg':
            self.scheduler = None # willa djust directly
        elif config.scheduler == 'default':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                            self.optimizer, step_size=10, gamma=1/1.5
                        )
        else:
           self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                           self.optimizer,
                           milestones=[9],
                           gamma=0.1
                           )
        
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        # put data on device
        x_train = x_train
        y_train = y_train
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size


        print('### Training... ###')
        train_losses = []

        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Manually update or use scheduler from pytorch

            # update optimizer if pesg only at epoch 9
            if epoch == 10 and self.config.optim == 'pesg':
                self.optimizer.update_regularizer(decay_factor=10)
         
            
            ### YOUR CODE HERE
            losses = []
            total_loss = 0
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                x_batch = curr_x_train[self.config.batch_size * i:self.config.batch_size * (i + 1)]
                y_batch = curr_y_train[self.config.batch_size * i:self.config.batch_size * (i + 1)]
                idx = torch.tensor(shuffle_index[self.config.batch_size * i:self.config.batch_size * (i + 1)], dtype=torch.long, device=self.device)
        
                # Don't forget to use "parse_record" to perform data preprocessing.
                if self.config.dataset == 'cifar':
                    x_batch = parse_record(x_batch, training=True, dim=self.config.img_dim, channels=self.config.channels, seed=epoch*100+i)
                    x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                else:
                    x_batch = radio_process(x_batch, training=True, device=self.device, seed=epoch*100+i)

                # move to device tensor
                y_batch = torch.tensor(y_batch, dtype=torch.long, device=self.device)

                ### YOUR CODE HERE
                self.optimizer.zero_grad()

                # forward pass
                y_pred = self.network(x_batch)
                if self.config.loss == 'ap':
                    loss = self.loss(y_pred, y_batch, idx)
                else:
                    loss = self.loss(y_pred, y_batch)

                # calc loss
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                total_loss += loss.item()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss.item()), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Avg Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, np.mean(total_loss), duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)

            # step scheduler if not pesg
            if self.scheduler:
                self.scheduler.step()

            # append losses
            train_losses.append(losses)

        ### YOUR CODE HERE
        #Plot the curves
        return np.array(train_losses)

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        
        x = x
        accs = []
        rocs = []
        aucs = []
        losses = []

        print('### Test or Validation ###')
        with torch.no_grad():
            for checkpoint_num in checkpoint_num_list:
                checkpointfile = os.path.join(self.config.modeldir, str(self.config.config_id), 'model-%d.ckpt'%(checkpoint_num))
                self.load(checkpointfile)

                if self.config.dataset == 'cifar':
                    x_std = parse_record(x, training=False, dim=self.config.img_dim, channels=self.config.channels)
                    x_std = torch.tensor(x_std, dtype=torch.float32, device=self.device)
                else:
                    x_std = radio_process(x, training=False, device=self.device)


                y_proba = self.network(x_std)
                if self.config.loss == 'ce':
                    loss = self.loss(y_proba, torch.tensor(y, device=self.device))
                else:
                    loss = None
               

                if self.config.dataset == 'cifar':
                    y_pred = torch.argmax(y_proba, dim=1).cpu().numpy()
                    y_proba = F.softmax(y_proba)
                elif self.config.loss == 'ce':
                    y_pred = torch.argmax(y_proba, dim=1).cpu().numpy()
                    y_proba = F.softmax(y_proba)[:, 1]
                else:
                    y_pred = (y_proba > 0.5).cpu().numpy().astype(int)
                y_proba = y_proba.cpu().numpy()

                acc = accuracy_score(y, y_pred)
                rocauc = roc_auc_score(y, y_proba, multi_class='ovr')
                prauc = average_precision_score(y, y_proba)

                accs.append(acc)
                rocs.append(rocauc)
                aucs.append(prauc)
                if loss:
                    losses.append(loss.item())

                print('Test accuracy: {:.4f}'.format(acc))
                print('ROC_AUC: {:.4f}'.format(rocauc))
                print('PR_AUC: {:.4f}'.format(prauc))

            return accs, rocs, aucs, losses
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, str(self.config.config_id), 'model-%d.ckpt'%(epoch))
        os.makedirs(os.path.join(self.config.modeldir, str(self.config.config_id)), exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))