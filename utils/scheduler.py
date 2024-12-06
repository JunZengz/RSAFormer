import torch
import numpy as np
# from thop import profile
# from thop import clever_format
from scipy.ndimage import map_coordinates

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import *


def get_scheduler(opts, optimizer, train_loader):
    if opts.Train.Scheduler.type == 'CosineAnnealingLR':
        return eval(opts.Train.Scheduler.type)(optimizer=optimizer,
                                                T_max=opts.Train.Scheduler.T_max,
                                                eta_min=opts.Train.Scheduler.eta_min,
                                                verbose=opts.Train.Scheduler.verbose)
    elif opts.Train.Scheduler.type == 'PolyLr':
        return eval(opts.Train.Scheduler.type)(optimizer, gamma=opts.Train.Scheduler.gamma,
                                               minimum_lr=opts.Train.Scheduler.minimum_lr,
                                               max_iteration=len(
                                                   train_loader) * opts.Train.Scheduler.epochs,
                                               warmup_iteration=opts.Train.Scheduler.warmup_iteration)



class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs