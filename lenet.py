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
        return z*(z>0)

    def backward(z):
        return 1*(z>0)

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
        pass

class avg_pooling():
    def __init__(self):
        pass

    def forward(feature, size=2, stride=2):
        
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

    def backward(next_dz, feature, size=2, stride=2):

        #平均池化反向过程
        #:param next_dz：损失函数关于最大池化输出的损失
        #:param feature: 卷积层矩阵,形状(B,C,H,W)，B为batch_size，C为通道数
        #:param pooling: 池化大小(k1,k2)
        #:param strides: 步长
        #:param padding: 0填充
        #:return:

        B, C, H, W = feature.shape
        _, _, out_h, out_w = next_dz.shape

        pool_dz = np.zeros((B, C, (out_h-1)*stride+size, (out_w-1)*stride+size), dtype=np.float32)

        for b in np.arange(B):
            for c in np.arange(C):
                for i in np.arange(out_h):
                    for j in np.arange(out_w):
                        # 每个神经元均分梯度
                        pool_dz[b, c,
                        stride * i:stride * i + size,
                        stride * j:stride * j + size] += next_dz[n, c, i, j] / (size*size)
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
        img_matrix = np.zeros((feature_h*feature_w, kernel_h*kernel_w*img_ch))
        kernel_matrix = np.zeros((kernel_h*kernel_w*img_ch, kernel_num))
        """
        将输入图片张量转换成矩阵形式
        """
        for j in range(img_ch):
            img_2d = np.copy(img[j,:,:])   
            shape=(feature_h,feature_w,kernel_h,kernel_w) 
            strides = (img_w,1,img_w,1)
            strides = img_2d.itemsize * np.array(strides)
            x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
            x_cols = np.ascontiguousarray(x_stride)
            x_cols = x_cols.reshape(feature_h*feature_w,kernel_h*kernel_w)
            img_matrix[:,j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w]=x_cols
        
        # 将输入图片张量转换成矩阵形式（另解）
        #for i in range(feature_h*feature_w):
        #    for j in range(img_ch):
        #        img_matrix[i, j*kernel_h*kernel_w:(j+1)*kernel_h*kernel_w] = img[np.uint16(i/feature_w):np.uint16(i/feature_w+kernel_h),np.uint16(i%feature_w):np.uint16(i%feature_w+kernel_w),j].reshape(kernel_h*kernel_w)    
    
        # 将卷积核张量转换成矩阵形式
        for i in range(kernel_num):
            kernel_matrix[:,i] = conv_kernel[i,:].transpose(2,0,1).reshape(kernel_w*kernel_h*img_ch) 
        
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
            nabla_b[i] = np.sum(out_img_delta[i,:,:])
        return nabla_b

    def add_bias(z, bias):
        for i in range(bias.shape[0]):
                z[i,:,:] += bias[i,0]
        return z

class loss():
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
        dy = y_probability - y_true
        return loss, dy

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
        dz = np.dot(next_dz, W.T)       # 当前层的梯度
        dw = np.dot(z.T, next_dz)       # 当前层权重的梯度
        db = np.sum(next_dz, axis=0)    # 当前层偏置的梯度, N个样本的梯度求和
        return dw / N, db / N, dz

def get_accuracy(pred, label, batch_size=16):
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
        scores[s:e] = LeNet.forward(pred[s:e])
    # 余数处理
    remain = num % batch_size
    if remain > 0:
        scores[-remain:] = LeNet.forward(pred[-remain:])
    # 计算精度
    accuracy = np.mean(np.argmax(scores, axis=1) == np.argmax(label, axis=1))
    return accuracy

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
        #print("forward started")

        B = x.shape[0]
        out = np.zeros([B , 10 , 1])
        
        for i in range(B):
            #第一次卷积
            conv1 = relu.forward(conv.add_bias(conv.forward(x[i,:,:,:] , self.kernels[0]), self.kernels_biases[0]))
            #print(conv1.shape)#(6, 24, 24)
            #print(conv1)
            pool1 = avg_pooling.forward(conv1)
            #print(pool1.shape)#(6, 12, 12)
            #print(pool1)

            #第二次卷积
            conv2 = relu.forward(conv.add_bias(conv.forward(pool1, self.kernels[1]), self.kernels_biases[1]))
            #print(conv2.shape)#(16, 8, 8)
            #print(conv2)
            pool2 = avg_pooling.forward(conv2)
            #print(pool2.shape)#(16, 4, 4)
            #print(pool2)

            #flatten
            fl = flatten.forward(pool2)
            #print(fl.shape)#(256, 1)
            #print(fl)

            #第一次FC
            fc1 = relu.forward(fc.forward(self.weights[0] , fl, self.biases[0]))
            #print(fc1.shape)#(128, 1)
            #print(fc1)

            #第二次FC
            fc2 = relu.forward(fc.forward(self.weights[1] , fc1, self.biases[1]))
            #print(fc2.shape)#(64, 1)
            #print(fc2)

            #第三次FC
            fc3 = relu.forward(fc.forward(self.weights[2] , fc2, self.biases[2]))
            #print(fc3.shape)#(10, 1)
            #print(fc3)

            #softmax
            softmax=soft_max.forward(fc3)
            #print(softmax.shape)#(10, 1)
            #print(softmax)
            out[i,:,:] = softmax
            #print(out.shape)#(16, 10, 1)
            #print(out)

        out=out.reshape(16, 10)
        #print(out.shape)#(16, 10)
        #print("forward finished")    
        return out

    def backward(self, error, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        print("backward started")

        for i in range(B):
            #softmax
            softmax=soft_max.backward(error)
            
            #第三次FC
            fc3 = fc.backward(relu.backward(softmax))
            
            #第二次FC
            fc2 = fc.backward(relu.backward(fc3))
            
            #第一次FC
            fc1 = fc.backward(relu.backward(fc2))
            
            #flatten
            fl = flatten.backward(fc1)
            
            #第二次卷积
            pool2 = avg_pooling.backward(fl)
            relu2 = relu.backward(pool2)
            conv2 = conv.backward(relu2)
            
            #第一次卷积
            pool1 = avg_pooling.backward(conv2)
            relu1 = relu.backward(pool1)
            conv1 = conv.backward(relu1)

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
        print("evaluatw started")
        acc=get_accuracy(x,labels)
        print("evaluatw finished")
        return acc

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images

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
    #        # 1. 一次forward，backward肯定不能是所有的图像一起,
    #        因此需要根据 batch_size 将 train_image 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
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
    #                print("Epoch %2d Iter %3d Loss %.5f" % (epoch, batch, loss))

    #            # 更新梯度
    #            #for w in ["W1", "b1", "W2", "b2", "W3", "b3"]:
    #            #    mnist.weights[w] -= lr * mnist.gradients[w]


    #        last = time.time() #计时开始
    #        '''for imgs, labels in zip(batch_images, batch_labels):
                
    #            这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
    #            我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
    #            2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
    #            3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
    #            4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                
    #            images = imgs.reshape(batch_size,1,28,28)
    #        #    print('yes')
    #            pred = self.forward(images)
    #            error = loss.cross_entropy_loss(pred,labels)
    #            iternum += 1
    #            print('iternum:%d' % iternum,"loss:",error)
    #        #    self.backward(batch_size,error)
    #        #    print('labels:',labels.shape)
    #            pass'''
    #        duration = time.time() - last
    #        sum_time += duration

    #        if epoch % 1 == 0:
    #            accuracy = self.evaluate(test_image, test_label)
    #            print("epoch{} accuracy{}".format(epoch, accuracy))
    #            accuracies.append(accuracy)

    #    avg_time = sum_time / epoches
    #    return avg_time, accuracies

    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 1,
        batch_size = 16,
        lr = 1.0e-3
    ):
        iternum = 0
        sum_time = 0
        accuracies = []
        print("fit started")
        last = time.time()
        #for epoch in range(epoches):

        #    train_image = self.data_augmentation(train_image)
            
        #    num_train = train_image.shape[0]
        #    num_batch = num_train // batch_size
        #    for batch in range(num_batch):
        #        # get batch data
        #        batch_mask = np.random.choice(num_train, batch_size, False)
        #        images = train_image[batch_mask]
        #        labels = train_label[batch_mask]
        #        images = images.reshape(batch_size,1,28,28)
        #        pred = self.forward(images)
        #        error,dy = loss.cross_entropy_loss(pred,labels)
        #        iternum += 1
        #        duration = time.time() - last
        #        last = time.time()
        #        speed = duration/batch_size
        #        print('iternum:%d' % iternum,"loss:",error,"speed:%.4f"% speed,"s/image")
        #        iternum += 0

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

        #if epoch % 1 == 0:
        accuracy = self.evaluate(test_image, test_label)
        print("epoch{} accuracy{}".format(epoch, accuracy))
        accuracies.append(accuracy)

        avg_time = sum_time / epoches
        print("fit finished")
        return avg_time, accuracies

def dnn_mnist():
    # load datasets
    path = 'mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    # 转为稀疏分类
    y_train, y_val,y_test =utils.to_categorical(y_train,10),utils.to_categorical(y_val,10),utils.to_categorical(y_test,10)

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