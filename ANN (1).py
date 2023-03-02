# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:47:37 2022

@author: duayb
"""

import tensorflow as tf
#ANN is defined.
class ANN():
    def __init__(self, num_input, locations, ann_config):
        ann = [a[0] for a in ann_config]
        dropout_rate = [a[1] for a in ann_config]

        self.X = tf.placeholder(tf.float32,
                                [None, num_input], 
                                name="input_X")
        self.Y = tf.placeholder(tf.float32, [None, locations], name="input_Y")

        Y = self.Y
        X = self.X
        features = X
        with tf.name_scope("ANN_part"):
            for idd, filter_size in enumerate(ann):
                with tf.name_scope("L"+str(idd)):
                    ann_out = tf.layers.dense(
                        features,
                        ann[idd],
                        name="ann_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer
                    )

            self.logits = tf.layers.dense(ann_out , 2,activation=tf.nn.relu)

        self.mse = tf.reduce_mean(tf.squared_difference(self.logits, self.Y))#Loss to be optimized.