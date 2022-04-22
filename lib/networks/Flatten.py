#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#Flatten
#打平层函数实现
from .utils import *
from .pool2row import *
from .Layer import *

__all__ = ['Flatten']

class Flatten:
    """
    flatten layer
    打平层，将卷积输出转换为全连接输入
    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        cache = inputs.shape
        return inputs.reshape(cache[0],-1), cache

    def backward(self, inputs, cache):
        return inputs.reshape(cache)

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved