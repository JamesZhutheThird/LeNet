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
        sum_time = 0
        accuracies = []
        
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
                pred = self.forward(images)
                error = loss.cross_entropy_loss(pred,labels)
                iternum += 1
                print('iternum:%d' % iternum,"loss:",error)

            for imgs, labels in zip(batch_images, batch_labels):

                images = imgs.reshape(batch_size,1,28,28)
            #    print('yes')
                pred = self.forward(images)
                error = loss.cross_entropy_loss(pred,labels)
                iternum += 1
                print('iternum:%d' % iternum,"loss:",error)
            #    self.backward(batch_size,error)
            #    print('labels:',labels.shape)
                pass
            duration = time.time() - last
            sum_time += duration

            if epoch % 1 == 0:
                accuracy = self.evaluate(train_image, train_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies