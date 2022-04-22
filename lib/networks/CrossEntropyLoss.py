#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#CrossEntropyLoss
#交叉熵函数实现
from .pool2row import *

__all__ = ['Conv2d']

class CrossEntropyLoss(object):

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        # scores.shape == [N, score]
        # labels.shape == [N]
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)

        N = labels.shape[0]
        correct_probs = probs[range(N), labels]
        loss = -1.0 / N * np.sum(np.log(correct_probs))
        return loss, probs

    def backward(self, probs, labels):
        grad_out = probs
        N = labels.shape[0]

        grad_out[range(N), labels] -= 1
        return grad_out

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved