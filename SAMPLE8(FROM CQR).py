from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from lenet import LeNet


def normalize_image(images):
    ''' 对图像做归一化处理 '''
    images = images/float(255)
    images -= np.mean(images)
    return images

def one_hot_labels(labels):
    '''
    将labels 转换成 one-hot向量
    eg:  label: 3 --> [0,0,0,1,0,0,0,0,0,0]
    labels_one_hot = np.zeros(10)
    labels_one_hot[labels] = 1
    '''
    labels_one_hot=[]
    for i in range(len(labels)):
        label = np.zeros(10)
        label[labels[i]]=1
        labels_one_hot.append(label)

    return np.array(labels_one_hot)


def main():
    # image shape: N x H x W, pixel [0, 255]
    # label shape: N x 10
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    #plt.imshow(x_train[59999], cmap='gray')
    #plt.show()
    print(x_train.shape, x_train[0].max(), x_train[0].min()) #(60000, 28, 28) 255 0 5
    print(x_test.shape, x_test[0].max(), x_test[0].min()) #(10000, 28, 28) 255 0 7

    x_train = normalize_image(x_train)
    x_test = normalize_image(x_test)
    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)


    net = LeNet()

    net.fit(x_train, y_train, x_test, y_test, epoches=10, batch_size=16, lr=1e-3)

    accu = net.evaluate(x_test, labels=y_test)
    print("final accuracy {}".format(accu))


if __name__ == "__main__":
    main()


