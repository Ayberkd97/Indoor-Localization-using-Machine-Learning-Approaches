import tensorflow as tf
from ANN import ANN
import numpy as np
import pandas as pd

class NetManager():
    def __init__(self, num_input, locations, learning_rate, data,
                 max_step_per_action=3,
                 bathc_size=100):

        self.num_input = num_input
        self.locations = locations
        self.learning_rate = learning_rate
        self.data = data
        
        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size

    def get_reward(self, action, step, pre_loss):
        action = [action[0][0][x:x+1] for x in range(0, len(action[0][0]), 1)]#Hyperparameters for each neural layer is divided.
        

        with tf.Graph().as_default() as g:
            with g.container('experiment'+str(step)):
                model = ANN(self.num_input, self.locations, action)#ANN is defined here.
                loss_op = tf.reduce_mean(model.mse)#loss
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)#Adam optimizer
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)
                    
                    groups_in_file = self.data.groupby(['position_x','position_y'])
                    for epoch,group in enumerate(groups_in_file.groups):
                        print(f"Epoch: {epoch}")
                        #for each 3 epochs, changed dataset is feeded.

                        temp_filter_not = (self.data['position_x'] != group[0])&(self.data['position_y'] != group[1])
                        temp_filter = (self.data['position_x'] == group[0])&(self.data['position_y'] == group[1])
                        frequencies = self.data.loc[temp_filter].iloc[:,1].values
                        frequencies = pd.Series(frequencies).drop_duplicates().tolist()
    
                        for f in frequencies:
                            for i in range(3):
                                # Iterate the training data to run the training step.
                                temp_train = self.data.loc[(temp_filter_not)&(self.data['frequency'] == f)].iloc[:,3:]
                                temp_val= self.data.loc[(temp_filter)&(self.data['frequency'] == f)].iloc[:,3:]
                                locations_not = self.data.loc[(self.data['frequency'] == f)&(temp_filter_not)].iloc[:,0:2].values
                                locations_val = self.data.loc[(self.data['frequency'] == f)&(temp_filter)].iloc[:,0:2].values

                                num_tr_iter = int(len(temp_train) / self.bathc_size)
                                for iteration in range(num_tr_iter):
                                    start = iteration * self.bathc_size
                                    end = (iteration + 1) * self.bathc_size
                                    batch_x = temp_train[start:end]
                                    batch_y = locations_not[start:end]
                                    feed = {model.X: batch_x,
                                            model.Y: batch_y}
                                    _ = train_sess.run(train_op, feed_dict=feed)
                                    
                                    if iteration%100 == 0:
                                        loss, mse = train_sess.run(
                                            [loss_op, model.mse],
                                            feed_dict={model.X: batch_x,
                                               model.Y: batch_y})
                                        print("iter {0:3d}:\t Loss={1:.2f},\t mse={2:.01%}".format(iteration, loss, mse))
                    #From the test set loss, reward is given.
                    batch_x = temp_val
                    batch_y = locations_val   
                    loss, pred = train_sess.run(
                        [loss_op, model.logits],
                        feed_dict={model.X: batch_x,
                                   model.Y: batch_y})
                    print("loss and pre_loss:", loss, pre_loss)
                    if loss - pre_loss <= 0.0:
                        return np.abs(loss-pre_loss), loss
                    else:
                        return -1, loss
