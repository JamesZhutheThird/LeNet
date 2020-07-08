from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time



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

##激活函数
def relu(feature):
    return feature*(feature>0)

def relu_back(feature):  # 对relu函数的求导
    return 1*(feature>0)



##第一层卷积 img二维，kernel四维,其中 img_ch = 1
def conv1(img,kernel):
    #传参，定义特征尺寸
    img_h , img_w = img.shape
    kernel_num , kernel_h , kernel_w , img_ch = kernel.shape
    feature_h = img_h - kernel_h + 1
    feature_w = img_w - kernel_w + 1
    
    #初始化特征图
    img_out = np.zeros((feature_h, feature_w , kernel_num))
    img_matrix = np.zeros((feature_h*feature_w , kernel_h*kernel_w))
    kernel_matrix = np.zeros((kernel_h*kernel_w , kernel_num))
    
    #img2col
    for i in range(feature_h*feature_w):
        img_matrix[i , 0:kernel_h*kernel_w] = img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h) , np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w)].reshape(kernel_h*kernel_w)
    for i in range(kernel_num):
        kernel_matrix[:,i] = kernel[i,:].reshape(kernel_w*kernel_h)
    
    #卷积
    feature_matrix = np.dot(img_matrix , kernel_matrix)
    
    #按照特征图尺寸输出
    for i in range(kernel_num):
        img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h , feature_w)
    
    return img_out


##第二层卷积，img三维，kernel四维
def conv2(img,kernel):
    #传参，定义特征尺寸
    img_h , img_w , img_ch = img.shape
    kernel_num , kernel_h , kernel_w , img_ch = kernel.shape
    feature_h = img_h - kernel_h + 1
    feature_w = img_w - kernel_w + 1
    
    #初始化特征图
    img_out = np.zeros((feature_h, feature_w , kernel_num))
    img_matrix = np.zeros((feature_h*feature_w , kernel_h*kernel_w))
    kernel_matrix = np.zeros((kernel_h*kernel_w , kernel_num))
    
    #img2col
    for i in range(feature_h*feature_w):
        for j in range(img_ch):
            img_matrix[i , j*(kernel_h*kernel_w):(j+1)*kernel_h*kernel_w] = img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h) , np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w) , j].reshape(kernel_h*kernel_w)
    for i in range(kernel_num):
        kernel_matrix[:,i] = kernel[i,:].reshape(kernel_w*kernel_h*img_ch)
    
    #卷积
    feature_matrix = np.dot(img_matrix , kernel_matrix)
    
    #按照特征图尺寸输出
    for i in range(kernel_num):
        img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h , feature_w)
    
    return img_out
    

##池化
def avgpool(feature):
    #传参，定义输出尺寸
    feature_h, feature_w, feature_ch = feature.shape
    pool_h = feature_h/2
    pool_w = feature_w/2
    
    #初始化输出图像
    out = np.zeros(pool_h , pool_w , feature_ch)
    
    f = feature/4
    
    for i in range(feature_ch):
        for j in range(pool_h):
            for k in range(pool_w):
                out[j,k,i] += f[2*j , 2*k , i]+f[2*j+1 , 2*k , i]+f[2*j , 2*k+1 , i]+f[2*j+1 , 2*k+1 , i] 
                
    return out

        
    

#用于激活最后一层FC
def softmax(z):
    tmp = np.max(z)
    z -= tmp       #缩放每行的元素，避免溢出
    z = np.exp(z)
    tmp = np.sum(z)
    z /= tmp
    return z




##avgpool误差反传
def pool_delta_error_bp(pool_out_delta):
    ##输出初始化
    delta = np.zeros([pool_out_delta.shape[0]*2 , pool_out_delta.shape[1]*2 , pool_out_delta.shape[2]])
    
    for i in range(pool_out_delta.shape[-1]):
        for j in range(pool_out_delta.shape[0]):
            for k in range(pool_out_delta.shape[1]):
                delta[2*j,2*k,i] = pool_out_delta[j,k,i]/4
                delta[2*j+1,2*k,i] = pool_out_delta[j,k,i]/4
                delta[2*j,2*k+1,i] = pool_out_delta[j,k,i]/4
                delta[2*j+1,2*k+1,i] = pool_out_delta[j,k,i]/4
    
    return delta
    



class LeNet(object):
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''
        #两个卷积核
        self.kernels = [np.random.randn(6,5,5,1)]
        self.kernels.append(np.random.randn(16, 5, 5, 6))
        
        #三个权值矩阵
        self.weights = [np.random.randn(128,256)]
        self.weights.append(np.random.randn(64,128))
        self.weights.append(np.random.randn(10,64))
        
        print("initialize")

    def init_weight(self):
        pass

    ##前传
    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率

        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        x_bat = x.shape[0]
        out = np.zeros[x_bat , 10 , 1]
        
        for i in range(x_bat):
            #第一次卷积
            relu1 = relu(conv1(x[i,:,:] , self.kernels[0]))
            pool1 = avgpool(relu1)
            
            #第二次卷积
            relu2 = relu(conv2(pool1 , self.kernels[1]))
            pool2 = avgpool(relu2)
            
            #flatten
            %%%bug is here%%%
            flat = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)
            
            #第一次FC
            fc1 = relu(np.dot(self.weights[0] , flat))
            
            #第二次FC
            fc2 = relu(np.dot(self.weights[1] , fc1))
            
            #第三次FC
            %%%bug is here%%%
            fc3 = softmax(np.dot(self.weights[2] , fc2))
            
            out[i,:,:] = fc3
            
        return out
    
    ##计算误差
    def compute_loss(pred, labels):
        error = pred - labels
        
        return error
    
    
    def backward(self, error, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        pass

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
        pred_labels = self.forward(x)
        correct_num = 0
        
        for i in range (x.shape[0]):
            if labels[i][np.argmax(pred_labels[i])] == 1:
                correct_num += 1
        
        return correct_num/x.shape[0]
    

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
            
            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1

            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                
                pass
            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies



