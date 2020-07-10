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
            conv1[i,:,:,:] = relu.forward(conv.add_bias(conv.forward(x[i,:,:,:] , self.kernels[0]), self.kernels_biases[0]))
            #print(conv1.shape)#(6, 24, 24)
            #print(conv1)
            pool1[i,:,:,:] = avg_pooling.forward(conv1[i,:,:,:])
            #print(pool1.shape)#(6, 12, 12)
            #print(pool1)

            #第二次卷积
            conv2[i,:,:,:] = relu.forward(conv.add_bias(conv.forward(pool1[i,:,:,:], self.kernels[1]), self.kernels_biases[1]))
            #print(conv2.shape)#(16, 8, 8)
            #print(conv2)
            pool2[i,:,:,:] = avg_pooling.forward(conv2[i,:,:,:])
            #print(pool2.shape)#(16, 4, 4)
            #print(pool2)

            #flatten
            fl[i,:,:] = flatten.forward(pool2[i,:,:])
            #print(fl.shape)#(256, 1)
            #print(fl)

            #第一次FC
            fc1[i,:,:] = relu.forward(fc.forward(self.weights[0] , fl[i,:,:], self.biases[0]))
            #print(fc1.shape)#(128, 1)
            #print(fc1)

            #第二次FC
            fc2[i,:,:] = relu.forward(fc.forward(self.weights[1] , fc1[i,:,:], self.biases[1]))
            #print(fc2.shape)#(64, 1)
            #print(fc2)

            #第三次FC
            fc3[i,:,:] = relu.forward(fc.forward(self.weights[2] , fc2[i,:,:], self.biases[2]))
            #print(fc3.shape)#(10, 1)
            #print(fc3)

            #softmax
            softmax[i,:,:] = soft_max.forward(fc3[i,:,:])
            #print(softmax.shape)#(10, 1)
            #print(softmax)
            out[i,:,:] = softmax[i,:,:]
            #print(out.shape)#(16, 10, 1)
            #print(out)

        out = out.reshape(16, 10)
        #print(out.shape)#(16, 10)
        #print("forward finished")
        

        loss, error = loss_cal.cross_entropy_loss(out,labels)


        print("backward started")

        B = error.shape[0]

        for i in range(B):
            #softmax
            softmax_[i,:,:] = soft_max.backward(error[i,:,:])
            print(softmax_[i,:,:].shape)#(10, 1)
            print(softmax_)

            #第三次FC
            fc3_[i,:,:] = fc.backward(relu.backward(softmax_[i,:,:]),fc2[i,:,:])
            print(fc3_.shape)#(, 1)
            print(fc3_)

            #第二次FC
            fc2_[i,:,:] = fc.backward(relu.backward(fc3_[i,:,:]),fc1[i,:,:])
            print(fc2_.shape)#(, 1)
            print(fc2_)

            #第一次FC
            fc1_[i,:,:] = fc.backward(relu.backward(fc2_[i,:,:]),fl[i,:,:])
            print(fc1_.shape)#(, 1)
            print(fc1_)

            #flatten
            fl_[i,:,:,:] = flatten.backward(fc1_[i,:,:])
            
            #第二次卷积
            pool2_[i,:,:,:] = avg_pooling.backward(fl_[i,:,:,:])
            relu2_[i,:,:,:] = relu.backward(pool2_[i,:,:,:])
            conv2_[i,:,:,:] = conv.backward(relu2_[i,:,:,:])
            
            #第一次卷积
            pool1_[i,:,:,:] = avg_pooling.backward(conv2_[i,:,:,:])
            relu1_[i,:,:,:] = relu.backward(pool1_[i,:,:,:])
            conv1_[i,:,:,:] = conv.backward(relu1_[i,:,:,:])

        print("backward finished")
        return out


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
            softmax = soft_max.forward(fc3)
            #print(softmax.shape)#(10, 1)
            #print(softmax)
            out[i,:,:] = softmax
            #print(out.shape)#(16, 10, 1)
            #print(out)

        out = out.reshape(16, 10)
        #print(out.shape)#(16, 10)
        #print("forward finished")
        return out

    def backward(self, error, lr = 1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        print("backward started")

        B = error.shape[0]

        for i in range(B):
            #softmax
            softmax = soft_max.backward(error)
            print(softmax.shape)#(10, 1)
            print(softmax)

            #第三次FC
            fc3 = fc.backward(relu.backward(softmax))
            print(fc3.shape)#(, 1)
            print(fc3)

            #第二次FC
            fc2 = fc.backward(relu.backward(fc3))
            print(fc2.shape)#(, 1)
            print(fc2)

            #第一次FC
            fc1 = fc.backward(relu.backward(fc2))
            print(fc1.shape)#(, 1)
            print(fc1)

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
