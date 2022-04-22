#小臣子吃大橙子
#8th,Jul,2020
#15th,Jul,2020
#solver
#求解器器函数实现
import numpy as np
import time
from tqdm import tqdm

__all__ = ['Solver']

class Solver(object):

    def __init__(self, model, data, criterion, optimizer, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_scheduler = kwargs.pop('lr_scheduler', None)
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.reg = kwargs.pop('reg', 1e-3)
        self.use_reg = self.reg != 0
        self.print_every = kwargs.pop('print_every', 1)
        self.num_evaluate = 0
        
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('未识别参数: %s' % extra)

        self._reset()

    def _reset(self):
        self.current_epoch = 0
        self.best_train_acc = 0
        self.best_test_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def _step(self, X_batch, y_batch):
        scores = self.model.forward(X_batch)
        loss, probs = self.criterion.forward(scores, y_batch)
        if self.use_reg:
            for k in self.model.params.keys():
                if 'W' in k:
                    loss += 0.5 * self.reg * np.sum(self.model.params[k] ** 2)

        grad_out = self.criterion.backward(probs, y_batch)
        grad = self.model.backward(grad_out)
        if self.use_reg:
            for k in grad.keys():
                if 'W' in k:
                    grad[k] += self.reg * self.model.params[k]

        self.optimizer.step(grad)

        return loss

    def evaluate(self, X, y, num_samples = None, batch_size = 8):
        """
        精度测试，如果num_samples小于X长度，则从X中采样num_samples个图片进行检测
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        if N < batch_size:
            batch_size = N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        
        y_pred = []
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.forward(X[start:end])
            y_pred.extend(np.argmax(scores, axis=1))
        acc = np.mean(y_pred == y)
        
        """
        y_pred = []
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.forward(X[start:end])
            y_pred.extend(np.argmax(scores, axis=1))
            y_label.extend(np.argmax(y, axis=1))
        acc = np.mean(y_pred == y_label)
        """
        return acc

    def fit(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        sum_time_1=0
        sum_time_2=0
        for i in range(self.num_epochs):
            self.current_epoch = i + 1
            print('\n开始第 %d 轮训练' % (self.current_epoch))
            start_1 = time.time()
            total_loss = 0.
            self.model.train()

            for j in tqdm(range(iterations_per_epoch)):
            #for j in tqdm(range(100)):
                idx_start = j * self.batch_size
                idx_end = (j + 1) * self.batch_size
                X_batch = self.X_train[idx_start:idx_end]
                y_batch = self.y_train[idx_start:idx_end]
                total_loss += self._step(X_batch, y_batch)

            end_1 = time.time()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            avg_loss = total_loss / iterations_per_epoch
            self.loss_history.append(float('%.6f' % avg_loss))
            duration_1 = end_1 - start_1
            print('\n第 %d 轮训练完成 耗时 %.2fs 损失 %.6f' % (self.current_epoch, duration_1, avg_loss))
            sum_time_1 += duration_1

            if self.current_epoch % self.print_every == 0:
                self.num_evaluate += 1
                print('开始准确率测试')
                self.model.eval()
                start_2 = time.time()
                train_acc = self.evaluate(self.X_train, self.y_train, batch_size=self.batch_size)
                test_acc = self.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
                end_2 = time.time()
                self.train_acc_history.append(train_acc)
                self.test_acc_history.append(test_acc)
                duration_2 = end_2 - start_2
                print('准确率测试完成 耗时 %.2fs 训练集准确率 %.4f 测试集准确率 %.4f' % (duration_2, train_acc, test_acc))
                sum_time_2 += duration_2

                if test_acc >= self.best_test_acc and train_acc > self.best_train_acc:
                    self.best_train_acc = train_acc
                    self.best_test_acc = test_acc
                    self.best_params = dict()
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
            
        # At the end of training swap the best params into the model
        self.model.params = self.best_params

        return sum_time_1/self.num_epochs, sum_time_2/self.num_evaluate

#Copyright © 2019-2020 James Zhu Ⅲ
#All Rights Reserved