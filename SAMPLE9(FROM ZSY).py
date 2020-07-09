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

def Calculate_output_size(input_size,filter_size,stride):
    return (input_size - filter_size) / stride + 1

def get_patch(input_array,i,j,filter_width,filter_height,stride):
    '''
    获取卷积区域
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
	    input_array_conv = input_array[start_i : start_i + filter_height,start_j : start_j + filter_width]
	    return input_array_conv

    elif input_array.ndim == 3:
        input_array_conv = input_array[:,start_i : start_i + filter_height,start_j : start_j + filter_width]
        return input_array_conv
    
    '''
    start_i = i * stride
    start_j = j * stride
    input_array_conv = input_array[:,start_i:start_i+filter_height,start_j:start_j+filter_width]
    print("input_array_conv:",input_array_conv)
    return input_array_conv
    '''

def conv_calculate(input_array,kernel_array,output_array,stride,bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (    
                get_patch(input_array, i, j, kernel_width, 
                    kernel_height, stride) * kernel_array
                ).sum() + bias

def element_wise_op(array,op):
    for i in np.nditer(array,op_flags = ['readwrite']):
        i[...] = op(i)

def padding(input_array,zp):
    '''
    为数组增加zero_padding
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth,input_height+2*zp,input_width+2*zp))
            padded_array[:,zp:zp+input_height,zp:zp+input_width] = input_array
            return padded_array
        else:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height+2*zp,input_width+2*zp))
            padded_array[zp:zp+input_height,zp:zp+input_width]
            return padded_array

class ReLu():
    def forward(self,input):
        return max(0,input)
    def backward(self,output):
        return 1 if output > 0 else 0

class Filter(object):
    def __init__(self,width,height,depth):
        self.weights = np.random.randn(depth,height,width) * 1e-2 #初始卷积核权重
        self.bias = 0 #初始偏置
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0
    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def update(self,learning_rate):
        self.weights -= learning_rate *  self.weights_grad
        self.bias -= learning_rate * self.bias_grad

class ConvLayer(object):
    '''
    参数含义：     
    input_width:输入图片尺寸——宽度
    input_height:输入图片尺寸——长度
    channel_number:通道数，彩色为3，灰色为1
    filter_width:卷积核的宽
    filter_height:卷积核的长
    filter_number:卷积核数量
    stride:步长
    activator:激活函数(ReLu)
    learning_rate:学习率
    output_width(height) 输出的宽和长
    output_array 总输出(channel,width,height)
    ''' 
    def __init__(self,input_width,input_height,channel_number,filter_width,
                filter_height,filter_number,zero_padding,stride,activator,learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate
        self.output_width = self.Calculate_output_size(self.input_width,filter_width,zero_padding,stride)
        self.output_height = self.Calculate_output_size(self.input_height,filter_height,zero_padding,stride)
        self.output_array = np.zeros((self.filter_number,self.output_height,self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,filter_height,channel_number))
    def forward(self,input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter_ = self.filters[f]
            conv_calculate(self.padded_input_array,filter_.get_weights(),self.output_array[f],self.stride,filter_.get_bias())
            element_wise_op(self.output_array,self.activator.forward)
    def backward(self,input_array,sensitivity_array,activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array,activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter_ in self.filters:
            filter_.update(self.learning_rate)
    
    def create_delta_array(self):
        return np.zeros((self.channel_number,self.input_height,self.input_width))
    
    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = int((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        padded_array = padding(expanded_array,zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter_ = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.zeros((filter_.get_weights().shape))
            for i in range(flipped_weights.shape[0]):
                flipped_weights[i] = np.rot90(filter_.get_weights()[i],2)
        #    flipped_weights = np.array(map(lambda i: np.rot90(i,2),filter_.get_weights()))
        #    print('fshape:',filter_.get_weights().shape)
        #    print('check',np.array(map(lambda i: np.rot90(i,2),filter_.get_weights())))
        #    print('fweights',filter_.get_weights())
        #    print('flipped',flipped_weights)
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv_calculate(padded_array[f],flipped_weights[d],delta_array[d],1,0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,activator.backward)
        self.delta_array *= derivative_array
    
    def bp_gradient(self,sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter_ = self.filters[f]
            for d in range(filter_.weights.shape[0]):
                conv_calculate(self.padded_input_array[d],expanded_array[f],filter_.weights_grad[d],1,0)
            # 计算偏置项的梯度
            filter_.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self,sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2* self.zero_padding + 1)
        # 构建新的sensitivity_map
        expanded_array = np.zeros((depth,expanded_height,expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expanded_array[:,i_pos,j_pos] = sensitivity_array[:,i,j]
        return expanded_array
    def Calculate_output_size(self,input_size,filter_size,zero_padding,stride):
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

class AvgPoolingLayer(object):
    def __init__(self,input_width,input_height,channel_number,filter_width,filter_height,stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / stride + 1)
        self.output_height = int((input_height - filter_height) / stride + 1)
        self.output_array = np.zeros((self.channel_number,self.output_height,self.output_width))

    def forward(self,input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (get_patch(input_array[d],i,j,self.filter_width,self.filter_height,self.stride).mean())

    def backward(self,input_array,sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    for w in range(self.filter_width):
                        for h in range(self.filter_height):
                            self.delta_array[d,i*self.stride+h,j*self.stride+w] = sensitivity_array[d,i,j] / (self.filter_width * self.filter_height)


class FCLayer(object):
    def __init__(self,input_size,output_size,learning_rate,activator):
    #    self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-1e-4,1e-4,(input_size,output_size))
        self.activator = activator
        self.bias = np.zeros((1,output_size))
        self.output_array = np.zeros((1,output_size))
    def forward(self,input_array):
        self.input_array = input_array
        self.output_array = input_array.dot(self.weights) + self.bias
        element_wise_op(self.output_array,self.activator.forward)
    def backward(self,input_array,sensitivity_array):
        self.delta_array = np.zeros((1,self.input_size))
        self.delta_array = sensitivity_array.dot(self.weights.T)
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,self.activator.backward)
        self.delta_array *= derivative_array
        self.grad_weights = self.input_array.T.dot(sensitivity_array) 
        self.grad_bias = np.sum(sensitivity_array,axis = 0)
        
    def update(self):
        self.weights -= self.grad_weights * self.learning_rate
        self.bias -= self.grad_bias * self.learning_rate

class SoftmaxLayer(object):
    def forward(self,input_array):
        self.input_array = input_array
        M = 0.0
    #    print('shapeXX',input_array.shape)
    #    print('shapeX:',self.input_array.shape)
    #    print('S:',self.input_array)
        for i in range(input_array.shape[1]):
            M = max(M,input_array[0][i])
        self.output_array = np.exp(input_array-M)
        self.sum_array = np.sum(self.output_array)
        self.output_array = self.output_array / self.sum_array
    def backward(self,input_array,sensitivity_array):
        self.forward(input_array)
        self.delta_array = self.output_array - 1.0
        self.delta_array = self.delta_array * sensitivity_array

## 在原 LeNet-5上进行少许修改后的 网路结构

'''
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
'''


class LeNet(object):
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''
        self.C1 = ConvLayer(28,28,1,5,5,6,0,1,ReLu(),1e-2)
        self.A1 = AvgPoolingLayer(24,24,6,2,2,2)
        self.C2 = ConvLayer(12,12,6,5,5,16,0,1,ReLu(),1e-2)
        self.A2 = AvgPoolingLayer(8,8,16,2,2,2)
        self.FC1 = FCLayer(256,128,1e-2,ReLu())
        self.FC1.weights = np.random.randn(256,128) * np.sqrt(2 / 256)
        self.FC2 = FCLayer(128,64,1e-2,ReLu())
        self.FC2.weights = np.random.randn(128,64) * np.sqrt(2 / 128)
        self.FC3 = FCLayer(64,10,1e-2,ReLu())
        self.FC3.weights = np.random.randn(64,10) * np.sqrt(2 / 64)
        self.S = SoftmaxLayer()
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
    #    print('x',x.shape)
    #    print('C1:',self.C1.filters[0].weights)
        self.results = np.zeros((x.shape[0],10))
        self.C1_input = np.zeros((x.shape))
        self.C1_gradient = np.zeros((x.shape[0],6,1,5,5))
        self.C1_bias_gradient = np.zeros((x.shape[0],6))
        self.A1_input = np.zeros((x.shape[0],6,24,24))
        self.C2_input = np.zeros((x.shape[0],6,12,12))
        self.C2_gradient = np.zeros((x.shape[0],16,6,5,5))
        self.C2_bias_gradient = np.zeros((x.shape[0],16))
        self.A2_input = np.zeros((x.shape[0],16,8,8))
        self.FC1_input = np.zeros((x.shape[0],256))
        self.FC1_gradient = np.zeros((x.shape[0],256,128))
        self.FC1_bias_gradient = np.zeros((x.shape[0],128))
        self.FC2_input = np.zeros((x.shape[0],128))
        self.FC2_gradient = np.zeros((x.shape[0],128,64))
        self.FC2_bias_gradient = np.zeros((x.shape[0],64))
        self.FC3_input = np.zeros((x.shape[0],64))
        self.FC3_gradient = np.zeros((x.shape[0],64,10))
        self.FC3_bias_gradient = np.zeros((x.shape[0],10))
        self.S_input = np.zeros((x.shape[0],10))
        for i in range(x.shape[0]):
            self.C1_input[i] = x[i]
            self.C1.forward(x[i])
            self.A1_input[i] = self.C1.output_array
            self.A1.forward(self.C1.output_array)
            self.C2_input[i] = self.A1.output_array
            self.C2.forward(self.A1.output_array)
            self.A2_input[i] = self.C2.output_array
            self.A2.forward(self.C2.output_array)
        #   print(self.A2.output_array)
            flatten_x = self.A2.output_array.reshape(1,-1)
            self.FC1_input[i] = flatten_x
            self.FC1.forward(flatten_x)
            self.FC2_input[i] = self.FC1.output_array
            self.FC2.forward(self.FC1.output_array.reshape(1,-1))
            self.FC3_input[i] = self.FC2.output_array
            self.FC3.forward(self.FC2.output_array.reshape(1,-1))
            self.S_input[i] = self.FC3.output_array
            self.S.forward(self.FC3.output_array.reshape(1,-1))
            self.results[i] = self.S.output_array
        #   print('sum',results[i].sum())
        #   print('for_shape',results.shape)
        #   print('for_data',results)
        return self.results

    def CrossEntropy(self,pred,labels):
        return -((labels * np.log10(pred+1e-7) + (1-labels) * np.log10(1-pred+1e-7))).sum() / pred.shape[0]

    def compute_loss(self,pred,labels):
        loss = 0
        self.sensitivity_array = np.zeros((pred.shape[0],10))
        self.sensitivity_array2 = np.zeros((pred.shape[0],10))
        for i in range(pred.shape[0]):
            self.sensitivity_array2[i] = pred[i] - labels[i]
            self.sensitivity_array[i] = labels[i] / (pred[i]+1e-5) + (1-labels[i]) / (1-pred[i]+1e-5)
            loss += self.CrossEntropy(pred[i],labels[i])
        self.sensitivity_array = np.mean(self.sensitivity_array,axis = 0)
        self.sensitivity_array = self.sensitivity_array.reshape(1,10)
        self.sensitivity_array2 = np.mean(self.sensitivity_array2,axis = 0)
        self.sensitivity_array2 = self.sensitivity_array2.reshape(1,10)
        loss /= pred.shape[0]
        return loss

    def backward(self, batch_size,error, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        # print('sensi:',self.sensitivity_array2)
        self.delta_array1 = self.sensitivity_array2
        self.delta_array2 = np.zeros((batch_size,64))
        self.delta_array3 = np.zeros((batch_size,128))
        self.delta_array4 = np.zeros((batch_size,256))
        self.delta_array5 = np.zeros((batch_size,16,8,8))
        self.delta_array6 = np.zeros((batch_size,6,12,12))
        self.delta_array7 = np.zeros((batch_size,6,24,24))
    #   print('shape1:',self.delta_array1.shape)
    #   print('check1:',self.delta_array1)
    #    for i in range(batch_size):
    #        self.S.backward(self.S_input[i].reshape(1,-1),self.sensitivity_array)
    #        self.delta_array1[i] = self.S.delta_array
    #    self.delta_array1 = np.mean(self.delta_array1,axis = 0)
    #    self.delta_array1 = self.delta_array1.reshape(1,-1)'''
    #   print('shape2:',self.delta_array1.shape)

        for i in range(batch_size):
            self.FC3.backward(self.FC3_input[i].reshape(1,-1),self.delta_array1)
            self.delta_array2[i] = self.FC3.delta_array
            self.FC3_gradient[i] = self.FC3.grad_weights
            self.FC3_bias_gradient[i] = self.FC3.grad_bias
        self.delta_array2 = np.mean(self.delta_array2,axis = 0)
        self.delta_array2 = self.delta_array2.reshape(1,-1)
        self.FC3_gradient = np.mean(self.FC3_gradient,axis = 0)
        self.FC3_bias_gradient = np.mean(self.FC3_bias_gradient,axis = 0)
        self.FC3.grad_weights = self.FC3_gradient
        self.FC3.grad_bias = self.FC3_bias_gradient
    #    print('FC3.1:',self.FC3.grad_weights[0])
        self.FC3.update()
    #    print('FC3_input[0]',self.FC3_input[0])
    #    print('FC3_grad:',self.FC3.grad_weights)
    #   print('shape3',self.delta_array2.shape)
    #   print('shape4',self.FC3_gradient.shape)
        for i in range(batch_size):
            self.FC2.backward(self.FC2_input[i].reshape(1,-1),self.delta_array2)
            self.delta_array3[i] = self.FC2.delta_array
            self.FC2_gradient[i] = self.FC2.grad_weights
            self.FC2_bias_gradient[i] = self.FC2.grad_bias
        self.delta_array3 = np.mean(self.delta_array3,axis = 0)
        self.delta_array3 = self.delta_array3.reshape(1,-1)
        self.FC2_gradient = np.mean(self.FC2_gradient,axis = 0)
        self.FC2_bias_gradient = np.mean(self.FC2_bias_gradient,axis = 0)
        self.FC2.grad_weights = self.FC2_gradient
        self.FC2.grad_bias = self.FC2_bias_gradient
        self.FC2.update()
    #   print('shape5',self.delta_array3.shape)
    #   print('shape6',self.FC2_gradient.shape)
        for i in range(batch_size):
            self.FC1.backward(self.FC1_input[i].reshape(1,-1),self.delta_array3)
            self.delta_array4[i] = self.FC1.delta_array
            self.FC1_gradient[i] = self.FC1.grad_weights
            self.FC1_bias_gradient[i] = self.FC1.grad_bias
        self.delta_array4 = np.mean(self.delta_array4,axis = 0)
        self.delta_array4 = self.delta_array4.reshape(1,-1)
        self.FC1_gradient = np.mean(self.FC1_gradient,axis = 0)
        self.FC1_bias_gradient = np.mean(self.FC1_bias_gradient,axis = 0)
        self.FC1.grad_weights = self.FC1_gradient
        self.FC1.grad_bias = self.FC1_bias_gradient
        self.FC1.update()
    #   print('shape7',self.delta_array4.shape)
    #   print('shape8',self.FC1_gradient.shape)
        self.delta_array4 = self.delta_array4.reshape(16,4,4)
        for i in range(batch_size):
            self.A2.backward(self.A2_input[i],self.delta_array4)
            self.delta_array5[i] = self.A2.delta_array
        self.delta_array5 = np.mean(self.delta_array5,axis = 0)
        self.delta_array5 = self.delta_array5.reshape(16,8,8)
    #   print('shape9',self.delta_array5.shape)
    #   print('shape10',self.C2.filters[0].weights_grad.shape)
    #   print('shape11',self.C2.filters[0].bias_grad.shape)
        for i in range(batch_size):
            self.C2.backward(self.C2_input[i],self.delta_array5,ReLu())
            self.delta_array6[i] = self.C2.delta_array
            for j in range(16):
                self.C2_gradient[i][j] = self.C2.filters[j].weights_grad
                self.C2_bias_gradient[i][j] = self.C2.filters[j].bias_grad
    #   print('shape10',self.C2.filters[0].weights_grad.shape)
    #   print('shape11',self.C2.filters[0].bias_grad.shape)
        self.C2_gradient = np.mean(self.C2_gradient,axis = 0)
        self.C2_bias_gradient = np.mean(self.C2_bias_gradient,axis = 0)
        self.C2_gradient = self.C2_gradient.reshape(16,6,5,5)
        self.C2_bias_gradient = self.C2_bias_gradient.reshape(16)
        for i in range(16):
            self.C2.filters[i].weights_grad = self.C2_gradient[i]
            self.C2.filters[i].bias_grad = self.C2_bias_gradient[i]
        self.C2.update()
        self.delta_array6 = np.mean(self.delta_array6,axis = 0)
        self.delta_array6 = self.delta_array6.reshape(6,12,12)
        for i in range(batch_size):
            self.A1.backward(self.A1_input[i],self.delta_array6)
            self.delta_array7[i] = self.A1.delta_array
        self.delta_array7 = np.mean(self.delta_array7,axis = 0)
        self.delta_array7 = self.delta_array7.reshape(6,24,24)
    #   print('shape12',self.delta_array7.shape)
        for i in range(batch_size):
            self.C1.backward(self.C1_input[i],self.delta_array7,ReLu())
            for j in range(6):
                self.C1_gradient[i][j] = self.C1.filters[j].weights_grad
                self.C1_bias_gradient[i][j] = self.C1.filters[j].bias_grad
    #   print('shape13',self.C1.filters[0].weights_grad.shape)
    #   print('shape14',self.C1.filters[0].bias_grad.shape)
        self.C1_gradient = np.mean(self.C1_gradient,axis = 0)
        self.C1_bias_gradient = np.mean(self.C1_bias_gradient,axis = 0)
        self.C1_gradient = self.C1_gradient.reshape(6,1,5,5)
        self.C1_bias_gradient = self.C1_bias_gradient.reshape(6)
     #   print('C1.1:',self.C1_gradient[0])
        for i in range(6):
            self.C1.filters[i].weights_grad = self.C1_gradient[i]
            self.C1.filters[i].bias_grad = self.C1_bias_gradient[i]
        self.C1.update()
    #    print('backward finish')

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
        count = 0.0
        pred = self.forward(x)
        for i in range(x.shape[0]):
            max_pos = 0
            max_p = 0
            for j in range(10):
                if max_p < pred[i][j]:
                    max_pos = j
                    max_p = pred[i][j]
            if(labels[i][max_pos] == 1):
                count += 1.0
        accu = count / x.shape[0]
        return accu

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
        epoches = 1, # old:10
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
            batch_size = 5
            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1
            tbatch_images = []
            tbatch_labels = []
            tmp_label = []
            tmp_image = []
            for i in range(int(train_image.shape[0]/batch_size)):
                if(i % 100 == 0): 
                    print("processing")
                for j in range(batch_size):
                    x = i*batch_size + j
                    tmp_image = train_image[x,:,:].reshape(1,28,28)
                    tmp_label = train_label[x,:].reshape(1,10)
                    if j == 0:
                        tbatch_images = tmp_image
                        tbatch_labels = tmp_label
                    else:
                        tbatch_images = np.concatenate((tbatch_images,tmp_image),axis = 0)
                        tbatch_labels = np.concatenate((tbatch_labels,tmp_label),axis = 0)
                tbatch_images = tbatch_images.reshape(1,batch_size,28,28)
                tbatch_labels = tbatch_labels.reshape(1,batch_size,10)
                if i == 0:
                    batch_images = tbatch_images
                    batch_labels = tbatch_labels
                else:
                    batch_images = np.concatenate((batch_images,tbatch_images),axis = 0)
                    batch_labels = np.concatenate((batch_labels,tbatch_labels),axis = 0)
        #    print('batch_images:',batch_images.shape)
        #    print('batch_labels:',batch_labels.shape)
            iternum = 0
            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                imgs_ = imgs.reshape(batch_size,1,28,28)
            #    print('yes')
                pred = self.forward(imgs_)
                error = self.compute_loss(pred,labels)
                iternum += 1
                print('iternum:%d' % iternum,"loss:",error)
                self.backward(batch_size,error)
            #    print('labels:',labels.shape)
            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies

def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    cl = ConvLayer(5,5,3,3,3,2,1,2,IdentityActivator(),0.001)
    cl.filters[0].weights = np.array(
        [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    cl.filters[0].bias=1
    cl.filters[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.forward(a)
    print ("前向传播结果:", cl.output_array)
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print ("反向传播后更新得到的filter1:",cl.filters[0])
    print ("反向传播后更新得到的filter2:",cl.filters[1])

class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1
