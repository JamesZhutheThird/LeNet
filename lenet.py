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

    def forward(z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))

class relu():
    def __init__(self):
        pass

    def forward(z):
        return z * (z > 0)

    def backward(z):
        return 1 * (z > 0)

class soft_max():
    def __init__(self):
        pass

    def forward(z):    
        tmp = np.max(z)
        #防止数值溢出
        z -= tmp
        z = np.exp(z)
        tmp = np.sum(z)
        z /= tmp
        return z

    def backward(z):
        #X, Y, Z = self.cache
        #dZ = np.zeros(X.shape)
        #dY = np.zeros(X.shape)
        #dX = np.zeros(X.shape)
        #N = X.shape[0]
        #for n in range(N):
        #    i = np.argmax(Z[n])
        #    dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
        #    M = np.zeros((N,N))
        #    M[:,i] = 1
        #    dY[n,:] = np.eye(N) - M
        #dX = np.dot(dout,dZ)
        #dX = np.dot(dX,dY)
        #return dX
        return z

class avg_pooling():
    def __init__(self):
        pass

    def forward(feature, size = 2, stride = 2):
        
        #平均池化前向过程
        #:param feature: 卷积层矩阵,形状(B,C,H,W)，N为Batch_size，C为通道数
        #:param pooling: 池化大小(k1,k2)
        #:param strides: 步长
        #:return:
        #print(feature.shape)#(6, 24, 24)
        C, H, W = feature.shape
        # 输出的高度和宽度
        out_h = (H - size) // stride + 1
        out_w = (W - size) // stride + 1

        pool_out = np.zeros((C, out_h, out_w), dtype=np.float32)


        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_out[c, i, j] = np.mean(feature[c,
                                                    stride * i:stride * i + size,
                                                    stride * j:stride * j + size])
        #print(pool_out.shape)
        return pool_out

    def backward(next_dz, feature, size = 2, stride = 2):

        #平均池化反向过程
        #:param next_dz：损失函数关于最大池化输出的损失
        #:param feature: 卷积层矩阵,形状(B,C,H,W)，B为batch_size，C为通道数
        #:param pooling: 池化大小(k1,k2)
        #:param strides: 步长
        #:param padding: 0填充
        #:return:

        C, H, W = feature.shape#(16, 8, 8)
        _, out_h, out_w = next_dz.shape#(16, 4, 4)

        pool_dz = np.zeros((C, (out_h - 1) * stride + size, (out_w - 1) * stride + size), dtype=np.float32)

        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 每个神经元均分梯度
                    pool_dz[c,
                    stride * i:stride * i + size,
                    stride * j:stride * j + size] += next_dz[c, i, j] / (size * size)
        return pool_dz

class flatten():
    def __init__(self):
        pass

    def forward(z):
        """
        将多维数组打平，前向传播
        :param z: 多维数组,形状(B,d1,d2,..)
        :return:
        """
        #B = z.shape[0]
        return z.reshape(-1,1)


    def backward(next_dz, z):
        """
        打平层反向传播
        :param next_dz:
        :param z:
        :return:
        """
        return next_dz.reshape(z.shape)

class conv():
    def __init__(self):
        pass

    def forward(img, conv_kernel):
      
        img_ch, img_h, img_w = img.shape
        kernel_num, kernel_h, kernel_w, img_ch = conv_kernel.shape
        feature_h = img_h - kernel_h + 1
        feature_w = img_w - kernel_w + 1

        # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
        img_out = np.zeros((feature_h, feature_w, kernel_num))
        img_matrix = np.zeros((feature_h * feature_w, kernel_h * kernel_w * img_ch))
        kernel_matrix = np.zeros((kernel_h * kernel_w * img_ch, kernel_num))
        """
        将输入图片张量转换成矩阵形式
        """
        for j in range(img_ch):
            img_2d = np.copy(img[j,:,:])   
            shape = (feature_h,feature_w,kernel_h,kernel_w) 
            strides = (img_w,1,img_w,1)
            strides = img_2d.itemsize * np.array(strides)
            x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
            x_cols = np.ascontiguousarray(x_stride)
            x_cols = x_cols.reshape(feature_h * feature_w,kernel_h * kernel_w)
            img_matrix[:,j * kernel_h * kernel_w:(j + 1) * kernel_h * kernel_w] = x_cols
        
        # 将输入图片张量转换成矩阵形式（另解）
        #for i in range(feature_h*feature_w):
        #    for j in range(img_ch):
        #        img_matrix[i, j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w] =
        #        img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h),np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w),j].reshape(kernel_h*kernel_w)
    
        # 将卷积核张量转换成矩阵形式
        for i in range(kernel_num):
            kernel_matrix[:,i] = conv_kernel[i,:].transpose(2,0,1).reshape(kernel_w * kernel_h * img_ch) 
        
        #卷积
        feature_matrix = np.dot(img_matrix, kernel_matrix)
        #print(feature_matrix.shape)#(576, 6)#(64, 16)
        #print(feature_matrix)

        #按照特征图尺寸输出
        for i in range(kernel_num):
            img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h , feature_w)

        #print(img_out.shape)
        img_out = img_out.transpose(2,0,1)
        #print(img_out.shape)

        return img_out

    def conv_cal_w(out_img_delta, in_img):
        # 同样利用img2col思想加速
        img_h, img_w, img_ch = in_img.shape
        feature_h, feature_w, kernel_num = out_img_delta.shape
        kernel_h = img_h - feature_h + 1
        kernel_w = img_w - feature_w + 1
    
        in_img_matrix = np.zeros([kernel_h * kernel_w * img_ch, feature_h * feature_w])
        out_img_delta_matrix = np.zeros([feature_h * feature_w, kernel_num])
    
        # 将输入图片转换成矩阵形式
        for j in range(img_ch):
            img_2d = np.copy(in_img[:,:,j])   
            shape = (kernel_h,kernel_w,feature_h,feature_w) 
            strides = (img_w,1,img_w,1)
            strides = img_2d.itemsize * np.array(strides)
            x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
            x_cols = np.ascontiguousarray(x_stride)
            x_cols = x_cols.reshape(kernel_h * kernel_w,feature_h * feature_w)
            in_img_matrix[j * kernel_h * kernel_w:(j + 1) * kernel_h * kernel_w,:] = x_cols
        # 将输入图片张量转换成矩阵形式（另解）
        for i in range(feature_h * feature_w):
            for j in range(img_ch):
                img_matrix[i, j * kernel_h * kernel_w:(j + 1) * kernel_h * kernel_w] = img[np.uint16(i / feature_w):np.uint16(i / feature_w + kernel_h),np.uint16(i % feature_w):np.uint16(i % feature_w + kernel_w),j].reshape(kernel_h * kernel_w)     
    
        # 将输出图片delta误差转换成矩阵形式
        for i in range(kernel_num):
            out_img_delta_matrix[:, i] = out_img_delta[:, :, i].reshape(feature_h * feature_w)
        
        kernel_matrix = np.dot(in_img_matrix, out_img_delta_matrix)
        nabla_conv = np.zeros([kernel_num, kernel_h, kernel_w, img_ch])
    
        for i in range(kernel_num):
            nabla_conv[i,:] = kernel_matrix[:,i].reshape(img_ch, kernel_h, kernel_w).transpose(1,2,0)

        return nabla_conv

    def conv_cal_b(out_img_delta):
        nabla_b = np.zeros((out_img_delta.shape[-1],1))
        for i in range(out_img_delta.shape[-1]):
            nabla_b[i] = np.sum(out_img_delta[i,:,:])
        return nabla_b

    def add_bias(z, bias):
        for i in range(bias.shape[0]):
                z[i,:,:] += bias[i,0]
        return z

    def conv_forward_bak(z, K, b, strides=(1, 1)):
        """
        多通道卷积前向过程
        :param z: 卷积层矩阵,形状(C,H,W),C为通道数
        :param K: 卷积核,形状(D,k1,k2,C),C为输入通道数,D为输出通道数
        :param b: 偏置,形状(D,)
        :param strides: 步长
        :return: 卷积结果
        """

        #N, _, height, width = padding_z.shape
        #C, D, k1, k2 = K.shape
        #assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
        #assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
        #conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
        #for n in np.arange(N):
        #    for d in np.arange(D):
        #        for h in np.arange(height - k1 + 1)[::strides[0]]:
        #            for w in np.arange(width - k2 + 1)[::strides[1]]:
        #                conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(
        #                    padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    


        _, height, width = z.shape
        D, k1, k2, C = K.shape
        print(z.shape)#(1, 28, 28)(6, 12, 12)(16, 8, 8)(6, 12, 12)
        print(K.shape)#(6, 5, 5, 1)(16, 5, 5, 6)(16, 5, 5, 6)(1, 8, 8, 16)
        kt = K.transpose(0,3,1,2)
        print(kt.shape)#(6, 1, 5, 5)(16, 6, 5, 5)(16, 6, 5, 5)(1, 16, 8, 8)

        conv_z = np.zeros((D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
        print(conv_z.shape)#(6, 24, 24)(16, 8, 8)

        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[d, h // strides[0], w // strides[1]] = np.sum(
                        z[:, h:h + k1, w:w + k2] * kt[d, :]) + b[d]
        
        print(conv_z.shape)#(6, 24, 24)

        return conv_z

    def conv_backward(next_dz, K, z, strides=(1, 1)):
        """
        多通道卷积层的反向过程
        :param next_dz: 卷积输出层的梯度,(D,H,W),H,W为卷积输出层的高度和宽度
        :param K: 当前层卷积核，(C,k1,k2,D)
        :param z: 卷积层矩阵,形状(C,H,W)，C为通道数
        :param strides: 步长
        :return:
        """

        _, H, W = z.shape
        D, k1, k2, C = K.shape
        D1, H1, W1 = next_dz.shape
        print(z.shape)#(6, 12, 12)
        print(K.shape)#(16, 5, 5, 6)(6, 5, 5, 1)


        # 卷积核梯度
        #dk = np.zeros((D, k1, k2, C))
        

        # 卷积核高度和宽度翻转180度
        flip_K = np.flip(K, (1, 2))
        swap_flip_K = np.swapaxes(flip_K, 0, 3)
        print(next_dz.shape)#(16, 8, 8)
        print(swap_flip_K.shape)#(6, 5, 5, 16)
        dz = conv.conv_forward_bak(next_dz, swap_flip_K, np.zeros((C,), dtype=np.float))

        # 求卷积核的梯度dK
        next_dz_=np.zeros([C, D1, H1, W1])
        for c in range(C):
            next_dz_[c,:,:,:] = next_dz
        next_dz_ = next_dz_.transpose(1,2,3,0)
        print(z.shape)#(6, 12, 12)
        print(next_dz_.shape)#(16, 8, 8, 6)
        dk = conv.conv_forward_bak(z, next_dz_, np.zeros((D,), dtype=np.float))
        print(z.shape)#(6, 12, 12)
        print(next_dz_.shape)#(16, 8, 8, 6)

        # 偏置的梯度
        dbk = np.sum(np.sum(next_dz, axis=-1), axis=-1)  # 在高度、宽度上相加；批量大小上相加

        print(dk.shape)#(16, 5, 5)
        print(dbk.shape)#(16,)
        print(dz.shape)#(6, 4, 4)

        return dk, dbk, dz

class loss_cal():
    def __init__(self):
        pass

    def mean_squared_loss(y_predict, y_true):
        """
        均方误差损失函数
        :param y_predict: 预测值,shape (N,d)，N为批量样本数
        :param y_true: 真实值
        :return:
        """
        loss = np.mean(np.sum(np.square(y_predict - y_true), axis=-1))  # 损失函数值
        dy = y_predict - y_true  # 损失函数关于网络输出的梯度
        return loss, dy

    def cross_entropy_loss(y_predict, y_true):
        """
        交叉熵损失函数
        :param y_predict: 预测值,shape (N,d)，N为批量样本数
        :param y_true: 真实值,shape(N,d)
        :return:
        """

        y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
        y_exp = np.exp(y_shift)
        y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
        loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
        error = y_probability - y_true
        return loss, error

class fc():
    def __init__(self):
        pass

    def forward(W, z, b):
        """
        全连接层的前向传播
        :param z: 当前层的输出,形状 (N,ln)
        :param W: 当前层的权重
        :param b: 当前层的偏置
        :return: 下一层的输出
        """
        return np.dot(W, z) + b


    def backward(next_dz, W, z):
        """
        全连接层的反向传播
        :param next_dz: 下一层的梯度
        :param W: 当前层的权重
        :param z: 当前层的输出
        :return:
        """
        N = z.shape[0]
        print(next_dz.shape)#(10, 1)(64, 1)
        print(W.shape)#(10, 64)(64, 128)
        print(z.shape)#(64, 1)(128, 1)
        dw = np.dot(next_dz, z.T)       # 当前层权重的梯度
        print(dw.shape)#(10, 64)(64, 128)
        db = np.sum(next_dz, axis=0)    # 当前层偏置的梯度, N个样本的梯度求和
        print(db.shape)#(1,)(1,)
        dz = np.dot(W.T, next_dz)       # 当前层的梯度
        print(dz.shape)#(64, 1)(128, 1)
        return dw / N, db / N, dz



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
        print("initialize started")
        
        #两个卷积核
        self.kernels = [np.random.randn(6, 5, 5, 1)] #图像变成 28*28*6 池化后图像变成14*14*6
        self.kernels_biases = [np.random.randn(6,1)]
        self.kernels.append(np.random.randn(16, 5, 5, 6)) #图像变成 10*10*16 池化后变成5*5*16
        self.kernels_biases.append(np.random.randn(16,1))
        
        #三个权值矩阵
        self.weights = [np.random.randn(128,256)]
        self.biases = [np.random.randn(128,1)]
        self.weights.append(np.random.randn(64,128))
        self.biases.append(np.random.randn(64,1))
        self.weights.append(np.random.randn(10,64))
        self.biases.append(np.random.randn(10,1))

        
        print("initialize finished")

    def init_weight(self):
        pass

    def forward(self, x):
        """
        前向传播
        x是训练样本，shape是B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率
        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        print("forward started")

        B = x.shape[0]
       
        
        self.conv1 = np.zeros([B,6,24,24])
        self.pool1 = np.zeros([B,6,12,12])
        self.conv2 = np.zeros([B,16,8,8])
        self.pool2 = np.zeros([B,16,4,4])
        self.fl = np.zeros([B,256,1])
        self.fc1 = np.zeros([B,128,1])
        self.fc2 = np.zeros([B,64,1])
        self.fc3 = np.zeros([B,10,1])
        self.softmax = np.zeros([B,10,1])
        self.out = np.zeros([B,10,1])



        for i in range(B):
            #第一次卷积
            #self.conv1[i,:,:,:] = relu.forward(conv.add_bias(conv.forward(x[i,:,:,:] , self.kernels[0]), self.kernels_biases[0]))
            self.conv1[i,:,:,:] = relu.forward(conv.conv_forward_bak(x[i,:,:,:] , self.kernels[0], self.kernels_biases[0]))
            print(self.conv1.shape)#(16, 6, 24, 24)
            #print(self.conv1)
            self.pool1[i,:,:,:] = avg_pooling.forward(self.conv1[i,:,:,:])
            #print(self.pool1.shape)#(16, 6, 12, 12)
            #print(self.pool1)

            #第二次卷积
            #self.conv2[i,:,:,:] = relu.forward(conv.add_bias(conv.forward(self.pool1[i,:,:,:], self.kernels[1]), self.kernels_biases[1]))
            self.conv2[i,:,:,:] = relu.forward(conv.conv_forward_bak(self.pool1[i,:,:,:] , self.kernels[1], self.kernels_biases[1]))
            print(self.conv2.shape)#(16, 16, 8, 8)
            #print(self.conv2)
            self.pool2[i,:,:,:] = avg_pooling.forward(self.conv2[i,:,:,:])
            #print(self.pool2.shape)#(16, 16, 4, 4)
            #print(self.pool2)

            #flatten
            self.fl[i,:,:] = flatten.forward(self.pool2[i,:,:])
            #print(self.fl.shape)#(16, 256, 1)
            #print(self.fl)

            #第一次FC
            self.fc1[i,:,:] = relu.forward(fc.forward(self.weights[0] , self.fl[i,:,:], self.biases[0]))
            #print(self.fc1.shape)#(16, 128, 1)
            #print(self.fc1)

            #第二次FC
            self.fc2[i,:,:] = relu.forward(fc.forward(self.weights[1] , self.fc1[i,:,:], self.biases[1]))
            #print(fc2.shape)#(16, 64, 1)
            #print(fc2)

            #第三次FC
            self.fc3[i,:,:] = relu.forward(fc.forward(self.weights[2] , self.fc2[i,:,:], self.biases[2]))
            #print(self.fc3.shape)#(16, 10, 1)
            #print(self.fc3)

            #softmax
            self.softmax[i,:,:] = soft_max.forward(self.fc3[i,:,:])
            #print(self.softmax.shape)#(16, 10, 1)
            #print(self.softmax)
            self.out[i,:,:] = self.softmax[i,:,:]
            #print(self.out.shape)#(16, 16, 10, 1)
            #print(self.out)

        self.out = self.out.reshape(16, 10)
        #print(out.shape)#(16, 10)
        print("forward finished")
        return self.out

    def backward(self, error, lr = 1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        print("backward started")

        B = error.shape[0]

        #self.conv1_ = np.zeros([B,6,24,24])
        #self.pool1_ = np.zeros([B,6,12,12])
        #self.conv2_ = np.zeros([B,16,8,8])
        #self.pool2_ = np.zeros([B,16,4,4])
        #self.fl_ = np.zeros([B,256,1])
        #self.fc1_ = np.zeros([B,128,1])
        #self.fc2_ = np.zeros([B,64,1])
        #self.fc3_ = np.zeros([B,10,1])
        #self.softmax_ = np.zeros([B,10,1])
        #self.out_ = np.zeros([B,10,1])

        self.conv1_ = np.zeros([B,6,28,28])
        self.pool1_ = np.zeros([B,6,24,24])
        self.conv2_ = np.zeros([B,6,12,12])
        self.pool2_ = np.zeros([B,16,8,8])
        self.dk1_=np.zeros([B,6,5,5])
        self.dk2_=np.zeros([B,16,5,5])
        self.dkb1_=np.zeros([B,1])
        self.dkb2_=np.zeros([B,1])
        self.fl_ = np.zeros([B,16,4,4])
        self.fc1_ = np.zeros([B,256,1])
        self.fc2_ = np.zeros([B,128,1])
        self.fc3_ = np.zeros([B,64,1])
        self.dw1_=np.zeros([B,128,256])
        self.dw2_=np.zeros([B,64,128])
        self.dw3_=np.zeros([B,10,64])
        self.db1_=np.zeros([B,1])
        self.db2_=np.zeros([B,1])
        self.db3_=np.zeros([B,1])
        self.softmax_ = np.zeros([B,10,1])
        self.out_ = np.zeros([B,10,1])
        



        error = error.reshape(16, 10, 1)

        for i in range(B):
            #softmax
            self.softmax_[i,:,:] = soft_max.backward(error[i,:,:])
            print(self.softmax_.shape)#(10, 1)
            #print(self.softmax_)

            #第三次FC
            self.dw3_[i,:,:],self.db3_[i,:],self.fc3_[i,:,:] = fc.backward(relu.backward(self.softmax_[i,:,:]),self.weights[2],self.fc2[i,:,:])
            print(self.dw3_.shape)#(16, 10, 64)
            #print(self.dw3_)
            print(self.db3_.shape)#(16, 1)
            #print(self.db3_)
            print(self.fc3_.shape)#(16, 64, 1)
            #print(self.fc3_)



            #第二次FC
            self.dw2_[i,:,:],self.db2_[i,:],self.fc2_[i,:,:] = fc.backward(relu.backward(self.fc3_[i,:,:]),self.weights[1],self.fc1[i,:,:])
            print(self.dw2_.shape)#(16, 64, 128)
            #print(self.dw2_)
            print(self.db2_.shape)#(16, 1)
            #print(self.db2_)
            print(self.fc2_.shape)#(16, 128, 1)
            #print(self.fc2_)

            #第一次FC
            self.dw1_[i,:,:],self.db1_[i,:],self.fc1_[i,:,:] = fc.backward(relu.backward(self.fc2_[i,:,:]),self.weights[0],self.fl[i,:,:])
            print(self.dw1_.shape)#(16, 128, 256)
            #print(self.dw1_)
            print(self.db1_.shape)#(16, 1)
            #print(self.db1_)
            print(self.fc1_.shape)#(16, 256, 1)
            #print(self.fc1_)

            #flatten
            self.fl_[i,:,:,:] = flatten.backward(self.fc1_[i,:,:], self.pool2[i,:,:,:])
            print(self.fl_.shape)#(16, 16, 4, 4)
            print(self.fl_)
            
            #第二次卷积
            self.pool2_[i,:,:,:] = avg_pooling.backward(self.fl_[i,:,:,:], self.conv2[i,:,:,:])
            self.dk2_[i,:,:,:],self.dkb2_[i,:],self.conv2_[i,:,:,:] = conv.conv_backward(relu.backward(self.pool2_[i,:,:,:]), self.kernels[1], self.pool1[i,:,:,:])
            
            x=np.zeros([1,28,28])

            #第一次卷积
            self.pool1_[i,:,:,:] = avg_pooling.backward(self.dk2_[i,:,:,:], self.conv1[i,:,:,:])
            self.dk1_[i,:,:,:],self.dkb1_[i,:],self.conv1_[i,:,:,:] = conv.conv_backward(relu.backward(self.pool1_[i,:,:,:]), self.kernels[0],x)

        print("backward finished")
        return out


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
        print("evaluate started")
        acc = self.get_accuracy(x, labels)
        print("evaluate finished")
        return acc

    def get_accuracy(self, pred, label, batch_size = 16):
        """
        :param pred:
        :param label:
        :param batch_size:
        :return:
        """
        scores = np.zeros_like(label, dtype=np.float)
        num = pred.shape[0]
        for i in range(num // batch_size):
            s = i * batch_size
            e = s + batch_size
            scores[s:e,:] = self.forward(pred[s:e,:,:,:])
        # 余数处理
        remain = num % batch_size
        if remain > 0:
            scores[-remain:,:] = self.forward(pred[-remain:,:,:,:])
        # 计算精度
        accuracy = np.mean(np.argmax(scores, axis=1) == np.argmax(label, axis=1))
        return accuracy

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images

    def fit(self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 1,
        batch_size = 16,
        lr = 1.0e-3):
        iternum = 0
        sum_time = 0
        accuracies = []
        print("fit started")
        last = time.time()

        for epoch in range(epoches):

            train_image = self.data_augmentation(train_image)
            
            num_train = train_image.shape[0]
            num_batch = num_train // batch_size
            
            for batch in range(num_batch):
                # get batch data
                batch_mask = np.random.choice(num_train, batch_size, False)
                images = train_image[batch_mask]
                labels = train_label[batch_mask]
                images = images.reshape(batch_size,1,28,28)
                pred = self.forward(images)
                loss, error = loss_cal.cross_entropy_loss(pred,labels)
                self.backward(error)
                iternum += 1
                duration = time.time() - last
                sum_time += duration
                last = time.time()
                speed = duration / batch_size
                print('iternum:%d' % iternum,"loss:",loss,"speed:%.4f" % speed,"s/image")
                iternum +=0


            #for imgs, labels in zip(batch_images, batch_labels):

            #    images = imgs.reshape(batch_size,1,28,28)
            ##    print('yes')
            #    pred = self.forward(images)
            #    error = loss.cross_entropy_loss(pred,labels)
            #    iternum += 1
            #    print('iternum:%d' % iternum,"loss:",error)
            ##    self.backward(batch_size,error)
            ##    print('labels:',labels.shape)
            #    pass
            #duration = time.time() - last
            #sum_time += duration

            if epoch % 1 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        print("fit finished")
        return avg_time, accuracies


    #def fit(
    #    self,
    #    train_image,
    #    train_label,
    #    test_image = None,
    #    test_label = None,
    #    epoches = 1,
    #    batch_size = 16,
    #    lr = 1.0e-3
    #):
    #    sum_time = 0
    #    accuracies = []

    #    for epoch in range(epoches):

    #        ## 可选操作，数据增强
    #        train_image = self.data_augmentation(train_image)
    #        ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应
    #        '''
    #        # 1.  一次forward，backward肯定不能是所有的图像一起,
    #        因此需要根据 batch_size 将 train_image 和 train_label 分成: [ batch0 |
    #        batch1 | ...  | batch_last]
    #        '''
    #        batch_images = [] # 请实现 step #1
    #        batch_labels = [] # 请实现 step #1
            
    #        num_train = X_train.shape[0]
    #        num_batch = num_train // batch_size
    #        for batch in range(num_batch):
    #            # get batch data
    #            batch_mask = np.random.choice(num_train, batch_size, False)
    #            X_batch = X_train[batch_mask]
    #            y_batch = y_train[batch_mask]
    #            # 前向及反向
    #            #mnist.forward(X_batch)
    #            #loss = mnist.backward(X_batch, y_batch)
    #            if batch % 200 == 0:
    #                print("Epoch %2d Iter %3d Loss %.5f" % (epoch, batch,
    #                loss))

    #            # 更新梯度
    #            #for w in ["W1", "b1", "W2", "b2", "W3", "b3"]:
    #            # mnist.weights[w] -= lr * mnist.gradients[w]


    #        last = time.time() #计时开始
    #        '''for imgs, labels in zip(batch_images, batch_labels):
                
    #            这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
    #            我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
    #            2.  做一次forward，得到pred结果 eg.  pred = self.forward(imgs)
    #            3.  pred 和 labels做一次 loss eg.  error = self.compute_loss(pred,
    #            labels)
    #            4.  做一次backward， 更新网络权值 eg.  self.backward(error, lr=1e-3)
                
    #            images = imgs.reshape(batch_size,1,28,28)
    #        # print('yes')
    #            pred = self.forward(images)
    #            error = loss.cross_entropy_loss(pred,labels)
    #            iternum += 1
    #            print('iternum:%d' % iternum,"loss:",error)
    #        # self.backward(batch_size,error)
    #        # print('labels:',labels.shape)
    #            pass'''
    #        duration = time.time() - last
    #        sum_time += duration

    #        if epoch % 1 == 0:
    #            accuracy = self.evaluate(test_image, test_label)
    #            print("epoch{} accuracy{}".format(epoch, accuracy))
    #            accuracies.append(accuracy)

    #    avg_time = sum_time / epoches
    #    return avg_time, accuracies
def dnn_mnist():
    # load datasets
    path = 'mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    # 转为稀疏分类
    y_train, y_val,y_test = utils.to_categorical(y_train,10),utils.to_categorical(y_val,10),utils.to_categorical(y_test,10)

    # bookeeping for best model based on validation set
    best_val_acc = -1
    mnist = Mnist()

    # Train
    batch_size = 32
    lr = 1e-1
    for epoch in range(10):
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        for batch in range(num_batch):
            # get batch data
            batch_mask = np.random.choice(num_train, batch_size, False)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            # 前向及反向
            mnist.forward(X_batch)
            loss = mnist.backward(X_batch, y_batch)
            if batch % 200 == 0:
                print("Epoch %2d Iter %3d Loss %.5f" % (epoch, batch, loss))

            # 更新梯度
            for w in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                mnist.weights[w] -= lr * mnist.gradients[w]

        train_acc = mnist.get_accuracy(X_train, y_train)
        val_acc = mnist.get_accuracy(X_val, y_val)

        if(best_val_acc < val_acc):
            best_val_acc = val_acc

        # store best model based n acc_val
        print('Epoch finish. ')
        print('Train acc %.3f' % train_acc)
        print('Val acc %.3f' % val_acc)
        print('-' * 30)
        print('')

    print('Train finished. Best acc %.3f' % best_val_acc)
    test_acc = mnist.get_accuracy(X_test, y_test)
    print('Test acc %.3f' % test_acc)