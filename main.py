#小臣子吃大橙子
#8th,Jul,2020
#15th,Jul,2020
#main
#LeNet主程序
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import lenet
import time
import lib
import lib.networks as net
import lib.ult as ult
from lib.ult import Draw

def normalize_image(images):
    ''' 对图像做归一化处理 '''
    #img = images.astype(np.float32)/127.5-1
    img = 2 * (images - np.min(images)) / (np.max(images) - np.min(images)) - 1
    return img

def formalize_matrix(z):
    out = z.reshape(z.shape[0],1,z.shape[1],z.shape[2])
    #out = z[:,np.newaxis,:,:]
    return out

def one_hot_labels(labels):
    '''
    将labels 转换成 one-hot向量
    eg:  label: 3 --> [0,0,0,1,0,0,0,0,0,0]
    '''
    lab = np.zeros((labels.size, 10))
    for i, row in enumerate(lab):
        row[labels[i]] = 1
    return lab

def main():
    print("小臣子吃大橙子のLeNet ver. 3.7.16")
    print("Copyright © 2019-2020 James Zhu Ⅲ")
    print("All Rights Reserved")

    TIME = time.time()

    print("\n\n-------开始读取预设参数-------")
    #time_2 = time.time()
    weight_scale1_ = 0.65
    weight_scale2_ = 0.025
    lr_ = 1e-3
    step_size_ = 1
    gamma_ = 0.75
    last_epoch_ = -1
    batch_size_ = 16
    num_epochs_ = 10
    print_every_ = 1
    print("卷积层权重范围    \t" , weight_scale1_ ,
        "\n全连接层权重范围  \t" , weight_scale2_,
        "\n学习率           \t", lr_,
        "\n学习率衰减步长    \t", step_size_, 
        "\n学习率衰减率      \t", gamma_, 
        "\n学习率衰减终止轮数\t", last_epoch_, 
        "\n批大小           \t", batch_size_, 
        "\n训练轮数          \t", num_epochs_, 
        "\n测试间隔轮数      \t",print_every_)
    #time_2 = time.time() - time_2
    #time_2 *= 1000
    #print("读取预设参数耗时  \t %.1fms" % time_2)
    print("-------预设参数读取完成-------")
    
    print("\n\n--------开始加载数据集--------")
    time_1 = time.time()
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    x_train = normalize_image(x_train)
    x_test = normalize_image(x_test)
    x_train = formalize_matrix(x_train)
    x_test = formalize_matrix(x_test)
    #y_train = one_hot_labels(y_train)
    #y_test = one_hot_labels(y_test)

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test
    }
    time_1 = time.time() - time_1
    print("加载数据集耗时    \t %.2fs" % time_1)
    print("--------数据集加载完成--------")

    print("\n\n-------开始初始化 LeNet-------")
    time_3 = time.time()
    model = lenet.LeNet(weight_scale1 = weight_scale1_, weight_scale2 = weight_scale2_)
    criterion = net.CrossEntropyLoss()
    optimizer = ult.SGD(model.params, lr = lr_)
    lr_scheduler_ = ult.lr_scheduler.StepLR(optimizer, step_size = step_size_, gamma = gamma_, last_epoch = last_epoch_)
    #fit函数详见ult.solver
    solver = ult.Solver(model, data, criterion, optimizer, lr_scheduler = lr_scheduler_, batch_size = batch_size_, num_epochs = num_epochs_ ,print_every = print_every_)
    time_3 = time.time() - time_3
    time_3 *= 1000
    print("初始化耗时        \t %.1fms" % time_3)
    print("-------LeNet 初始化完成-------")

    print("\n\n--------开始训练 LeNet--------")
    time_4, time_5 = solver.fit()   
    print("\n--------LeNet 训练完成--------")

    TIME = time.time() - TIME

    print('总运行时长 %.2fs 平均训练耗时 %.2fs 平均测试耗时 %.2fs' % (TIME, time_4, time_5))
    print('最佳训练集准确率 %.4f 最佳测试集准确率 %.4f' % (solver.best_train_acc, solver.best_test_acc))
    print("\n小臣子吃大橙子のLeNet ver. 3.7.16")
    print("Copyright © 2019-2020 James Zhu Ⅲ")
    print("All Rights Reserved\n")
    plt = Draw()
    plt.multi_plot((solver.train_acc_history, solver.test_acc_history, solver.loss_history), ('train', 'test', 'loss'),
                    title='训练结果', xlabel='迭代/次', ylabel='损失        准确率')
    
    

if __name__ == "__main__":
    main()

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved