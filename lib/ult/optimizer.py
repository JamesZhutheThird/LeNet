#小臣子吃大橙子
#8th,Jul,2020
#15th,Jul,2020
#optimizer
#优化器抽象类定义
from abc import ABCMeta, abstractmethod

__all__ = ['Optimizer']

class Optimizer(metaclass=ABCMeta):

    def __init__(self, params, defaults):
        self.params = params

        self.optim_configs = {}
        for p in params.keys():
            # d = {k: v for k, v in defaults.items()}
            d = defaults.copy()
            self.optim_configs[p] = d

    @abstractmethod
    def step(self, grad):
        pass

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved