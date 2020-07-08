from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import math
from functools import reduce


# Example Sigmoid
# 这个类中包含了 forward 和backward函数
class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))




## 在原 LeNet-5上进行少许修改后的 网路结构
"""
conv1: in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2: in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1: in_channel: 256, out_channels: 128, activation: relu
fc2: in_channel: 128, out_channels: 64, activation: relu
fc3: in_channel: 64, out_channels: 10, activation: relu
softmax:

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
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''     
        class Relu(object):
            def __init__(self, shape):
                self.eta = np.zeros(shape)
                self.x = np.zeros(shape)
                self.output_shape = shape

            def forward(self, x):
                self.x = x
                return np.maximum(x, 0)

            def gradient(self, eta):
                self.eta = eta
                self.eta[self.x<0]=0
                return self.eta

        class Conv2D(object):
            def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
                self.input_shape = shape
                self.output_channels = output_channels
                self.input_channels = shape[-1]
                self.batchsize = shape[0]
                self.stride = stride
                self.ksize = ksize
                self.method = method

                weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
                self.weights = np.random.standard_normal(
                    (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
                self.bias = np.random.standard_normal(self.output_channels) / weights_scale

                if method == 'VALID':
                    self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) / self.stride, (shape[1] - ksize + 1) / self.stride,
                    self.output_channels))

                if method == 'SAME':
                    self.eta = np.zeros((shape[0], shape[1]/self.stride, shape[2]/self.stride,self.output_channels))

                self.w_gradient = np.zeros(self.weights.shape)
                self.b_gradient = np.zeros(self.bias.shape)
                self.output_shape = self.eta.shape

                if (shape[1] - ksize) % stride != 0:
                    print ('input tensor width can\'t fit stride')
                if (shape[2] - ksize) % stride != 0:
                    print ('input tensor height can\'t fit stride')

            def forward(self, x):
                col_weights = self.weights.reshape([-1, self.output_channels])
                if self.method == 'SAME':
                    x = np.pad(x, (
                        (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                                    'constant', constant_values=0)
            
                self.col_image = []
                conv_out = np.zeros(self.eta.shape)
                for i in range(self.batchsize):
                    img_i = x[i][np.newaxis, :]
                    self.col_image_i = im2col(img_i, self.ksize, self.stride)
                    conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
                    self.col_image.append(self.col_image_i)
                self.col_image = np.array(self.col_image)
                return conv_out

            def backward(self, alpha=0.00001, weight_decay=0.0004):
                 # weight_decay = L2 regularization
                 self.weights *= (1 - weight_decay)
                 self.bias *= (1 - weight_decay)
                 self.weights -= alpha * self.w_gradient
                 self.bias -= alpha * self.bias
                 self.w_gradient = np.zeros(self.weights.shape)
                 self.b_gradient = np.zeros(self.bias.shape)

            def gradient(self, eta):
                self.eta = eta
                col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

                for i in range(self.batchsize):
                    self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
                self.b_gradient += np.sum(col_eta, axis=(0, 1))

        def im2col(image, ksize, stride):
            # image is a 4d tensor([batchsize, width ,height, channel])
            image_col = []
            for i in range(0, image.shape[1] - ksize + 1, stride):
                for j in range(0, image.shape[2] - ksize + 1, stride):
                    col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                    image_col.append(col)
            image_col = np.array(image_col)

            return image_col


        if __name__ == "__main__":
            # img = np.random.standard_normal((2, 32, 32, 3))
            img = np.ones((1, 32, 32, 3))
            img *= 2
            conv = Conv2D(img.shape, 12, 3, 1)
            next = conv.forward(img)
            next1 = next.copy() + 1
            conv.gradient(next1-next)
            conv.backward()

        class AvgPooling(object):
            def __init__(self, shape, ksize=2, stride=2):
                self.input_shape = shape
                self.ksize = ksize
                self.stride = stride
                self.output_channels = shape[-1]
                self.integral = np.zeros(shape)
                self.index = np.zeros(shape)
                self.output_shape = [shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels]


            def gradient(self, eta):
                # stride = ksize
                next_eta = np.repeat(eta, self.stride, axis=1)
                next_eta = np.repeat(next_eta, self.stride, axis=2)
                next_eta = next_eta*self.index
                return next_eta/(self.ksize*self.ksize)

            def forward(self, x):
                for b in range(x.shape[0]):
                    for c in range(self.output_channels):
                        for i in range(x.shape[1]):
                            row_sum = 0
                            for j in range(x.shape[2]):
                                row_sum += x[b, i, j, c]
                                if i == 0:
                                    self.integral[b, i, j, c] = row_sum
                                else:
                                    self.integral[b, i, j, c] = self.integral[b, i - 1, j, c] + row_sum

                out = np.zeros([x.shape[0], x.shape[1] / self.stride, x.shape[2] / self.stride, self.output_channels],
                            dtype=float)

                # integral calculate pooling
                for b in range(x.shape[0]):
                    for c in range(self.output_channels):
                        for i in range(0, x.shape[1], self.stride):
                            for j in range(0, x.shape[2], self.stride):
                                self.index[b, i:i + self.ksize, j:j + self.ksize, c] = 1
                                if i == 0 and j == 0:
                                    out[b, i / self.stride, j / self.stride, c] = self.integral[
                                        b, self.ksize - 1, self.ksize - 1, c]

                                elif i == 0:
                                    out[b, i / self.stride, j / self.stride, c] = self.integral[b, 1, j + self.ksize - 1, c] - \
                                                                                self.integral[b, 1, j - 1, c]
                                elif j == 0:
                                    out[b, i / self.stride, j / self.stride, c] = self.integral[b, i + self.ksize - 1, 1, c] - \
                                                                                self.integral[b, i - 1, 1, c]
                                else:
                                    out[b, i / self.stride, j / self.stride, c] = self.integral[
                                                                                    b, i + self.ksize - 1, j + self.ksize - 1, c] - \
                                                                                self.integral[
                                                                                    b, i - 1, j + self.ksize - 1, c] - \
                                                                                self.integral[
                                                                                    b, i + self.ksize - 1, j - 1, c] + \
                                                                                self.integral[b, i - 1, j - 1, c]

                out /= (self.ksize * self.ksize)
                return out

        
        class FullyConnect(object):
            def __init__(self, shape, output_num=2):
                self.input_shape = shape
                self.batchsize = shape[0]
                input_len = reduce(lambda x, y: x * y, shape[1:])
                self.weights = np.random.standard_normal((input_len, output_num))/100
                self.bias = np.random.standard_normal(output_num)/100

                self.output_shape = [self.batchsize, output_num]
                self.w_gradient = np.zeros(self.weights.shape)
                self.b_gradient = np.zeros(self.bias.shape)
            def forward(self, x):
                self.x = x.reshape([self.batchsize, -1])
                output = np.dot(self.x, self.weights)+self.bias
                return output

            def backward(self, eta):
                for i in range(eta.shape[0]):
                    col_x = self.x[i][:, np.newaxis]
                    eta_i = eta[i][:, np.newaxis].T
                    self.w_gradient += np.dot(col_x, eta_i)
                    self.b_gradient += eta_i.reshape(self.bias.shape)

                next_eta = np.dot(eta, self.weights.T)
                next_eta = np.reshape(next_eta, self.input_shape)

                return next_eta

        class Softmax(object):
            def __init__(self, shape):
                self.softmax = np.zeros(shape)
                self.eta = np.zeros(shape)
                self.batchsize = shape[0]

            def cal_loss(self, prediction, label):
                self.label = label
                self.prediction = prediction
                self.predict(prediction)
                self.loss = 0
                for i in range(self.batchsize):
                    self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

                return self.loss

            def predict(self, prediction):
                exp_prediction = np.zeros(prediction.shape)
                self.softmax = np.zeros(prediction.shape)
                for i in range(self.batchsize):
                    prediction[i, :] -= np.max(prediction[i, :])
                    exp_prediction[i] = np.exp(prediction[i])
                    self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
                return self.softmax
            def backward(self):
                self.eta = self.softmax.copy()
                for i in range(self.batchsize):
                    self.eta[i, self.label[i]] -= 1
                return self.eta

        self.conv1 = Conv2D([64,1,28,28], 6, 5, 1)
        self.relu1 = Relu(self.conv1.output_shape)
        self.pool1 = AvgPooling(self.relu1.output_shape)
        self.conv2 = Conv2D(self.pool1.output_shape, 16, 5, 1)
        self.relu2 = Relu(self.conv2.output_shape)
        self.pool2 = AvgPooling(self.relu2.output_shape)
        self.fc1 = FullyConnect(self.pool2.output_shape, 128)
        self.fc2 = FullyConnect(self.fc1.output_shape, 64)
        self.fc3 = FullyConnect(self.fc2.output_shape, 10)
        self.sf = Softmax(self.fc3.output_shape)
        self.p2_shape=None
        

    print("initialize")


    def init_weight(self):

        pass

    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率

        Arguments:
            x {np.array} --shape为 B，C，H，W
            
        """
        h1 = self.conv1.forward(x)
        a1 = self.relu1.forward(h1)
        p1 = self.pool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.relu2.forward(h2)
        p2 = self.pool2.forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(x.shape[0],-1) # Flatten 转化为列向量
        h3 = self.fc1.forward(fl)
        h4 = self.fc2.forward(h3)
        h5 = self.fc3.forward(h4)
        a5 = self.sf.predict(h5)

        return a5
        
        

    def backward(self, error, lr=1e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        self.sf.backward()
        self.conv1.backward(
        self.relu1.gradient(
        self.pool1.gradient(
        self.conv2.backward(
        self.relu2.gradient(
        self.pool2.gradient(
        self.fc1.backward(self.fc2.backward(self.fc3.backward(self.sf.eta))).reshape(self.p2_shape)))))))
    

    def evaluate(self, x, labels):
        """
        x是测试样本， shape 是BCHW
        labels是测试集中的标注， 为one-hot的向量
        返回的是分类正确的百分比

        在这个函数中，建议直接调用一次forward得到pred_labels,
        再与 labels 做判断

        Arguments:
            x {np array} -- BCWH
            labels {np array} -- B x 10
        """
    

        accuracy=0
        

        return accuracy

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images

    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 10,
        batch_size = 16,
        lr = 1.0e-3
    ):
        sum_time = 0
        accuracies = []

        

        for epoch in range(epoches):

            ## 可选操作，数据增强
            train_image = self.data_augmentation(train_image)
            ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应
            '''
            # 1. 一次forward，bachword肯定不能是所有的图像一起,
            因此需要根据 batch_size 将 train_image, 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
            '''
            batch_images = [train_image[:128],train_image[128:]] # 请实现 step #1
            batch_labels = [train_image[:128],train_image[128:]] # 请实现 step #1

            last = time.time() #计时开始

            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                #forward
                

                #backward
                
        

                


            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies


