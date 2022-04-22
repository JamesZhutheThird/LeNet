#小臣子吃大橙子
#8th,Jul,2020
#14th,Jul,2020
#accuracy
#计算准确率
import numpy as np

class Accuracy(object):

    def __init__(self):
        pass

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        predicted = np.argmax(scores, axis=1)
        return np.mean(predicted == labels), predicted

    def check_accuracy(self, data_array, labels_array, net, batch_size = 128):
        total_accuracy = 0.0
        num = 0

        range_list = np.arange(0, data_array.shape[0] - batch_size, step=batch_size)
        for i in range_list:
            data = data_array[i:i + batch_size]
            labels = labels_array[i:i + batch_size]

            scores = net.predict(data)
            predicted = np.argmax(scores, axis=1)
            total_accuracy += np.mean(predicted == labels)
            num += 1

        return total_accuracy / num
    
#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved