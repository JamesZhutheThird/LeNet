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
            def __init__(self, shape, output_channels, ksize=5, stride=1, method='VALID'):
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
                    shapes=(shape[0], int((shape[1] - ksize + 1) / self.stride), 
                    int((shape[1] - ksize + 1) / self.stride),
                    self.output_channels)
                    self.eta = np.zeros(shapes)

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
                # deconv of padded eta with flippd kernel to get next_eta
                if self.method == 'VALID':
                    pad_eta = np.pad(self.eta, (
                        (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                                    'constant', constant_values=0)
                if self.method == 'SAME':
                    pad_eta = np.pad(self.eta, (
                        (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                                    'constant', constant_values=0)
                flip_weights = np.flipud(np.fliplr(self.weights))
                flip_weights = flip_weights.swapaxes(2, 3)
                col_flip_weights = flip_weights.reshape([-1, self.input_channels])
                col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
                next_eta = np.dot(col_pad_eta, col_flip_weights)
                next_eta = np.reshape(next_eta, self.input_shape)
                return next_eta

        def im2col(image, ksize, stride):
            # image is a 4d tensor([batchsize, width ,height, channel])
            image_col = []
            for i in range(0, image.shape[1] - ksize + 1, stride):
                for j in range(0, image.shape[2] - ksize + 1, stride):
                    col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                    image_col.append(col)
            image_col = np.array(image_col)

            return image_col

        class AvgPooling(object):
            def __init__(self, shape, ksize=2, stride=1):
                self.input_shape = shape
                self.ksize = ksize
                self.stride = stride
                self.output_channels = shape[-1]
                self.integral = np.zeros(shape)
                self.index = np.zeros(shape)
                self.output_shape = [shape[0], shape[1] , shape[2] , self.output_channels]
                
            def backward(self, eta):
                next_eta = np.repeat(eta, self.stride, axis=1)
                next_eta = np.repeat(next_eta, self.stride, axis=2)
                next_eta = next_eta*self.index
                return next_eta/(self.ksize*self.ksize)

            def forward(self, x):
                out = np.zeros([x.shape[0], x.shape[1], x.shape[2] , self.output_channels])

                for b in range(x.shape[0]):
                    for c in range(self.output_channels):
                        for i in range(0, x.shape[1], self.stride):
                            for j in range(0, x.shape[2], self.stride):
                                out[b, i, j , c] = np.mean(
                                    x[b, i:i + self.ksize, j:j + self.ksize, c])
                                
                return out

        class FullyConnect(object):
            def __init__(self, shape, output_num=2):
                self.input_shape = shape
                self.batchsize = shape[0]
                input_len = reduce(lambda x, y: x * y, shape[1:])
                self.weights = np.random.standard_normal((int(input_len), output_num))/100
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
                """self.label = label
                self.prediction = prediction
                self.predict(prediction)
                self.loss = 0
                for i in range(self.batchsize):
                    self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]
                    #for x in range(10):
                        #if label[i][x]==1:
                            #self.loss += np.log(np.sum(prediction[i])) - np.log(prediction[i][x])

                return self.loss"""
                self.label = label
                self.prediction = prediction
                loss = 0.0
                N = prediction.shape[0]
                M = np.sum(prediction*label, axis=1)
                for e in M:
                    #print(e)
                    if e == 0:
                        loss += 500
                    else:
                        loss += -np.log(e)

                return loss/N

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
                    self.eta[i, np.argmax(self.label,axis=1)] -= 1
                return self.eta

        self.conv1 = Conv2D([16,28,28,1], 6, 5, 1)
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
        x[[1,3], :] = x[[1,3], :]
        x =x [:, : ,:,np.newaxis]
        h1 = self.conv1.forward(x)
        a1 = self.relu1.forward(h1)
        p1 = self.pool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.relu2.forward(h2)
        p2 = self.pool2.forward(a2)
        self.p2_shape = p2.shape
        #fl = p2.reshape(x.shape[0],-1) # Flatten 转化为列向量
        fl=p2.flatten()
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
        self.conv1.gradient(
        self.relu1.gradient(
        self.pool1.backward(
        self.conv2.gradient(
        self.relu2.gradient(
        self.pool2.backward(
        self.fc1.backward(
        self.fc2.backward(
        self.fc3.backward(self.sf.eta))).reshape(self.p2_shape)))))))
    

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
    
        x[[1,3], :] = x[[3,1], :]
        x =x [:, : ,:,np.newaxis]
        accuracy=0
        i=0
        sum=0

        net=LeNet()
        pred_labels=net.forward(x)
        pred_label=np.zeros((x.shape[0],10))


        for i in range(pred_labels.shape[0]):
            pred_label[i][np.argmax(pred_labels[i])]=1
            judge=(pred_label[i]==labels[i])
            if judge.all():
                sum=sum+1
            i=i+1

        accuracy=sum/pred_labels.shape[0]
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
            batch_images =[] # 请实现 step #1
            batch_labels =[] # 请实现 step #1

            for i in range(int(train_image.shape[0] / batch_size)):
                batch_images .append(train_image[i * batch_size:(i + 1) * batch_size]) # 请实现 step #1
                batch_labels .append(train_label[i * batch_size:(i + 1) * batch_size]) # 请实现 step #1

            last = time.time() #计时开始

            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                pred=self.forward(imgs)
                error=self.sf.cal_loss(pred,labels)
                self.backward(error,lr=1e-5)
                pass

            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies


