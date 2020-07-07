from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

# Example Sigmoid
# 这个类中包含了 forward 和backward函数
class sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

class relu():
    def __init__(self):
        pass

    def forward(self, x):
        return x*(x>0)

    def backward(self, x):
        return 1*(x>0)

def soft_max(self, x):    
        tmp = np.max(x)
        x -= tmp  
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
        return x

def conv(img, conv_kernel):
    """
    if len(img.shape)!=3 or len(conv_kernel.shape)!=4:
        print("卷积运算所输入的维度不符合要求")
        sys.exit()
        
    if img.shape[-1] != conv_kernel.shape[-1]:
        print("卷积输入图片与卷积核的通道数不一致")
        sys.exit()
    """   
    img_h, img_w, img_ch = img.shape
    kernel_num, kernel_h, kernel_w, img_ch = conv_kernel.shape
    feature_h = img_h - kernel_h + 1
    feature_w = img_w - kernel_w + 1

    # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
    img_out = np.zeros((feature_h, feature_w, kernel_num))
    img_matrix = np.zeros((feature_h*feature_w, kernel_h*kernel_w*img_ch))
    kernel_matrix = np.zeros((kernel_h*kernel_w*img_ch, kernel_num))
    """
    将输入图片张量转换成矩阵形式
    for j in range(img_ch):
        img_2d = np.copy(img[:,:,j])   
        shape=(feature_h,feature_w,kernel_h,kernel_w) 
        strides = (img_w,1,img_w,1)
        strides = img_2d.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
            x_cols = x_cols.reshape(feature_h*feature_w,kernel_h*kernel_w)
            img_matrix[:,j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w]=x_cols
    """
    # 将输入图片张量转换成矩阵形式（另解）
    for i in range(feature_h*feature_w):
        for j in range(img_ch):
            img_matrix[i, j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w] = img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h),np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w),j].reshape(kernel_h*kernel_w)    
    
    # 将卷积核张量转换成矩阵形式
    for i in range(kernel_num):
        kernel_matrix[:,i] = conv_kernel[i,:].transpose(2,0,1).reshape(kernel_w*kernel_h*img_ch) 

    feature_matrix = np.dot(img_matrix, kernel_matrix)

    for i in range(kernel_num):
        img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h, feature_w)

    return img_out

def conv_cal_w(out_img_delta, in_img):
    # 同样利用img2col思想加速
    img_h, img_w, img_ch = in_img.shape
    feature_h, feature_w, kernel_num = out_img_delta.shape
    kernel_h = img_h - feature_h + 1
    kernel_w = img_w - feature_w + 1
    
    in_img_matrix = np.zeros([kernel_h*kernel_w*img_ch, feature_h*feature_w])
    out_img_delta_matrix = np.zeros([feature_h*feature_w, kernel_num])
    
    # 将输入图片转换成矩阵形式
    for j in range(img_ch):
        img_2d = np.copy(in_img[:,:,j])   
        shape=(kernel_h,kernel_w,feature_h,feature_w) 
        strides = (img_w,1,img_w,1)
        strides = img_2d.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols = x_cols.reshape(kernel_h*kernel_w,feature_h*feature_w)
        in_img_matrix[j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w,:]=x_cols
    # 将输入图片张量转换成矩阵形式（另解）
    for i in range(feature_h*feature_w):
        for j in range(img_ch):
            img_matrix[i, j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w] = img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h),np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w),j].reshape(kernel_h*kernel_w)     
    
    # 将输出图片delta误差转换成矩阵形式
    for i in range(kernel_num):
        out_img_delta_matrix[:, i] = out_img_delta[:, :, i].reshape(feature_h*feature_w)
        
    kernel_matrix = np.dot(in_img_matrix, out_img_delta_matrix)
    nabla_conv = np.zeros([kernel_num, kernel_h, kernel_w, img_ch])
    
    for i in range(kernel_num):
        nabla_conv[i,:] = kernel_matrix[:,i].reshape(img_ch, kernel_h, kernel_w).transpose(1,2,0)

    return nabla_conv

def conv_cal_b(out_img_delta):
    nabla_b = np.zeros((out_img_delta.shape[-1],1))
    for i in range(out_img_delta.shape[-1]):
        nabla_b[i] = np.sum(out_img_delta[:,:,i])
    return nabla_b



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
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类，AvgPool类，Relu类，FC类对象，SoftMax类对象
        并给Conv类与FC类对象赋予随机初始值
        注意：不要求做BatchNormlize和DropOut，但是有兴趣的可以尝试
        '''
        self.kernels = [np.random.randn(6, 5, 5, 1)] #图像变成 28*28*6 池化后图像变成14*14*6
        self.kernels_biases = [np.random.randn(6,1)]
        self.kernels.append(np.random.randn(16, 5, 5, 6)) #图像变成 10*10*16 池化后变成5*5*16
        self.kernels_biases.append(np.random.randn(16,1))
        
        self.weights = [np.random.randn(128,256)]
        self.weights.append(np.random.randn(64,128))
        self.weights.append(np.random.randn(10,64))
        self.biases = [np.random.randn(128,1)]
        self.biases.append(np.random.randn(64,1))
        self.biases.append(np.random.randn(10,1))
        
        print("initialize")

    def init_weight(self):
        pass

    

    def forward(self, x):
        """前向传播
        x是训练样本，shape是B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率

        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        return 0

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
        return 0

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
            # 1. 一次forward，backward肯定不能是所有的图像一起,
            因此需要根据 batch_size 将 train_image 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
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


