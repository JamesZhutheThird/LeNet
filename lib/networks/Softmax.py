#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#Softmax
#归一化指数函数实现
import numpy as np

__all__ = ['Softmax']

class Softmax(object):
    """
    softmax评分
    """

    def __init__(self):
        pass

    def __call__(self, scores):
        return self.forward(scores)

    def forward(self, scores):
        # scores.shape == [N, C]
        assert len(scores.shape) == 2
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)

        return probs

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved