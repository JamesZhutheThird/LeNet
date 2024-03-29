#小臣子吃大橙子
#8th,Jul,2020
#15th,Jul,2020
#LRScheduler
#学习率迭代抽象类定义
from abc import ABCMeta, abstractmethod
from lib.ult import Optimizer

class LRScheduler(metaclass=ABCMeta):

    def __init__(self, optimizer, last_epoch = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.optim_configs.values():
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.optim_configs.values()):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.optim_configs.values()))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    @abstractmethod
    def get_lr(self):
        pass

    def step(self, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.optim_configs.values(), self.get_lr()):
            param_group['lr'] = lr

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved