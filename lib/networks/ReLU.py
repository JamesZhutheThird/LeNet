#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#ReLU
#线性整流函数实现
import numpy as np
from .Layer import *

__all__ = ['ReLU']

class ReLU(object):

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs, alpha = 0):
        return np.maximum(inputs, inputs * alpha)

    def backward(self, grad_out, inputs, alpha = 0):
        assert inputs.shape == grad_out.shape

        grad_in = grad_out.copy()
        grad_in[inputs < 0] *= alpha
        return grad_in

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved