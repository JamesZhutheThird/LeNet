#小臣子吃大橙子
#8th,Jul,2020
#15th,Jul,2020
#lenet
#LeNet具体函数实现
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import lib.networks as net
import lib.ult as ult

## 在原LeNet-5上进行少许修改后的网路结构
"""
conv1:    in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2:    in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1:      in_channel: 256, out_channels: 128, activation: relu
fc2:      in_channel: 128, out_channels: 64, activation: relu
fc3:      in_channel: 64, out_channels: 10, activation: relu
softmax

tensor: (1x28x28)   --conv1    -->  (6x24x24)
tensor: (6x24x24)   --avgpool1 -->  (6x12x12)
tensor: (6x12x12)   --conv2    -->  (16x8x8)
tensor: (16x8x8)    --avgpool2 -->  (16x4x4)
tensor: (16x4x4)    --flatten  -->  (256)
tensor: (256)       --fc1      -->  (128)
tensor: (128)       --fc2      -->  (64)
tensor: (64)        --fc3      -->  (10)
tensor: (10)        --softmax  -->  (10)
"""

class LeNet(object):
    def __init__(self, in_channels = 1, out_channels = 10, dropout = 1.0, weight_scale1 = 0.6, weight_scale2 = 0.1):
        '''
        初始化网路，在这里你需要，声明各Conv类，AvgPool类，Relu类，FC类对象，SoftMax类对象
        并给Conv类与FC类对象赋予随机初始值
        注意：不要求做BatchNormlize和DropOut，但是有兴趣的可以尝试
        '''
        
        self.conv1 = net.Conv2d(in_channels, 5, 5, 6, stride=1, padding=0, weight_scale=weight_scale1)
        self.conv2 = net.Conv2d(6, 5, 5, 16, stride=1, padding=0, weight_scale=weight_scale1)
        #self.maxPool1 = net.MaxPool(2, 2, 6, stride=2)
        #self.maxPool2 = net.MaxPool(2, 2, 16, stride=2)
        self.avgPool1 = net.AvgPool(2, 2, 6, stride=2)
        self.avgPool2 = net.AvgPool(2, 2, 16, stride=2)
        self.fc1 = net.FC(256, 128, weight_scale=weight_scale2)
        self.fc2 = net.FC(128, 64, weight_scale=weight_scale2)
        self.fc3 = net.FC(64, out_channels, weight_scale=weight_scale2)
        self.flatten = net.Flatten()
        self.relu = net.ReLU()
        self.sofxmax = net.Softmax()
        self.z1 = None
        self.z1_cache = None
        self.z2 = None
        self.z2_cache = None
        self.z3 = None
        self.z3_cache = None
        self.z4 = None
        self.z4_cache = None
        self.z5 = None
        self.z5_cache = None
        self.z6 = None
        self.z6_cache = None
        self.z7 = None
        self.z7_cache = None
        self.z8 = None
        self.z8_cache = None

        self.use_dropout = dropout != 1.0
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'
            self.dropout_param['p'] = dropout
            self.dropout = net.Dropout()
            self.U1 = None
            self.U2 = None

        self.params = self.init_weight()

    def __call__(self, inputs):
        return self.forward(inputs)

    def init_weight(self):
        params = dict()

        params['W1'], params['b1'] = self.conv1.get_params()
        params['W2'], params['b2'] = self.conv2.get_params()
        params['W3'], params['b3'] = self.fc1.get_params()
        params['W4'], params['b4'] = self.fc2.get_params()
        params['W5'], params['b5'] = self.fc3.get_params()
        return params

    def forward(self, inputs):
        """
        前向传播
        x是训练样本，shape是B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率
        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        # inputs.shape = [N, C, H, W]
        #print("forward started")
        assert len(inputs.shape) == 4

        self.z1, self.z1_cache = self.conv1(inputs, self.params['W1'], self.params['b1'])
        a1 = self.relu(self.z1)
        self.z2, self.z2_cache = self.avgPool1(a1)
        self.z3, self.z3_cache = self.conv2(self.z2, self.params['W2'], self.params['b2'])
        a3 = self.relu(self.z3)
        self.z4, self.z4_cache = self.avgPool2(a3)
        self.z5, self.z5_cache = self.flatten(self.z4)
        self.z6, self.z6_cache = self.fc1(self.z5, self.params['W3'], self.params['b3'])
        a6 = self.relu(self.z6)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U1 = self.dropout(a6, self.dropout_param)
            a6 *= self.U1
        self.z7, self.z7_cache = self.fc2(a6, self.params['W4'], self.params['b4'])
        a7 = self.relu(self.z7)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U2 = self.dropout(a6, self.dropout_param)
            a7 *= self.U2
        self.z8, self.z8_cache = self.fc3(a7, self.params['W5'], self.params['b5'])
        
        output = self.relu(self.z8)

        #print("forward finished")
        return output

    def backward(self, grad_out):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        #print("backward started")
        grad = dict()
        
        dz8 = self.relu.backward(grad_out, self.z8)
        grad['W5'], grad['b5'], da7 = self.fc3.backward(dz8, self.z8_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da7 *= self.U1
        dz7 = self.relu.backward(da7, self.z7)
        grad['W4'], grad['b4'], da6 = self.fc2.backward(dz7, self.z7_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da6 *= self.U1
        dz6 = self.relu.backward(da6, self.z6)
        grad['W3'], grad['b3'], da5 = self.fc1.backward(dz6, self.z6_cache)
        dz4 = self.flatten.backward(da5, self.z5_cache)
        da3 = self.avgPool2.backward(dz4, self.z4_cache)
        dz3 = self.relu.backward(da3, self.z3)
        grad['W2'], grad['b2'], da2 = self.conv2.backward(dz3, self.z3_cache)
        da1 = self.avgPool1.backward(da2, self.z2_cache)
        dz1 = self.relu.backward(da1, self.z1)
        grad['W1'], grad['b1'], da0 = self.conv1.backward(dz1, self.z1_cache)
        
        #print("backward finished")
        return grad

    #fit函数详见ult.solver

    def train(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'

    def eval(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'test'

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reservedfrom __future__ import absolute_import