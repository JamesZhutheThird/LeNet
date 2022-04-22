#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#Sigmoid
#S型激活函数实现（未调用）
import numpy as np
from .Layer import *

__all__ = ['Sigmoid']

class Sigmoid(object):

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        return 1.0 / (1.0 + np.exp(-inputs))

    def backward(self, inputs):
        return self.forward(inputs) * (1 - self.forward(inputs))

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved