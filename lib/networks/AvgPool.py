#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#AvgPool
#平均池化层函数实现
from .pool2row import *
from .Layer import *
from .utils import *

__all__ = ['AvgPool']

class AvgPool:
    """
    average pool layer
    平均池化层，执行average运算
    """

    def __init__(self, filter_h, filter_w, filter_num, stride = 2):
        super(AvgPool, self).__init__()
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_num = filter_num
        self.stride = stride

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h) / self.stride + 1)
        out_w = int((W - self.filter_w) / self.stride + 1)

        a = pool2row_indices(inputs, self.filter_h, self.filter_w, stride=self.stride)
        z = np.mean(a, axis=1)

        #arg_z = np.argmax(a, axis=1)
        input_shape = inputs.shape
        a_shape = a.shape
        #cache = (arg_z, input_shape, a_shape)
        cache = (input_shape, a_shape)

        return pool_fc2output(z, N, out_h, out_w), cache

    def backward(self, grad_out, cache):
        #arg_z, input_shape, a_shape = cache
        input_shape, a_shape = cache

        dz = pool_output2fc(grad_out)
        dz = dz.reshape(dz.shape[0],-1)
        da = np.ones(a_shape)
        da *= dz / (self.filter_h * self.filter_w)

        return row2pool_indices(da, input_shape, field_height=self.filter_h, field_width=self.filter_w,
                                stride=self.stride)

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reservedfrom .utils import *