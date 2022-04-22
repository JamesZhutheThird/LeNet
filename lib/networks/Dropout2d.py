#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#Dropout2d
#通道随机失活函数实现（未调用）
import numpy as np

class Dropout2d(object):

    def __call__(self, shape, p):
        return self.forward(shape, p)

    def forward(self, shape, p):
        assert len(shape) == 4
        N, C, H, W = shape[:4]
        U = (np.random.rand(N * C, 1) < p) / p
        res = np.ones((N * C, H * W))
        res *= U

        if np.sum(res) == 0:
            return 1.0 / p
        return res.reshape(N, C, H, W)

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved