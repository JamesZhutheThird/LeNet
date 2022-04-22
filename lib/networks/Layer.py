#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#Layer
#网络层抽象类定义
from abc import ABCMeta, abstractmethod

__all__ = ['Layer']

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out, cache):
        pass

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved