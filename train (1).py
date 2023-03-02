# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:43:29 2022

@author: duayb
"""

import numpy as np
import tensorflow as tf
import argparse
import datetime
import pandas as pd
import io

from ANN import ANN
from NetManager import NetManager
from Reinforce import Reinforce


def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)
    #Number of layers for algorithm to run the experiments is written here.
    parser.add_argument('--max_layers', default=3)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


def policy_network(state, max_layers):
    with tf.name_scope("policy_network"):#This context manager pushes a name scope
        nas_cell = tf.contrib.rnn.NASCell(2*max_layers)#Neural Architecture Search (NAS) recurrent network cell from the paper. The number of units in the NAS cell. We write 1 for just the neuron numbers.
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )#Creates a recurrent neural network specified by RNNCell cell
        bias = tf.Variable([0.05]*2*max_layers)#bias
        outputs = tf.nn.bias_add(outputs, bias)#adds bias
        print("outputs: ", outputs, outputs[:, -1:, :],  tf.slice(outputs, [0, 1*max_layers-1, 0], [1, 1, 1*max_layers]))# Returned last output of rnn
        return outputs[:, -1:, :]      

def train(data):
    global args
    sess = tf.Session()#A class for running TensorFlow operations.
    global_step = tf.Variable(0.1, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)#Applies exponential decay to the learning rate.
#decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
                        
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)#Optimization like Adam for reinforcement algorithm.

    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, global_step)#Reinforce algorithm is defined
    net_manager = NetManager(num_input=4,
                             locations=2,
                             learning_rate=0.001,
                             data=data,
                             bathc_size=100)#NetManager variables are defined.

    MAX_EPISODES = 100#Number of episode for Reinforce algorithm
    step = 0
    state = np.array([[20.0]*args.max_layers], dtype=np.float32)#Initialized value is defined. 
    pre_loss = 0.0
    total_rewards = 0
    for i_episode in range(MAX_EPISODES):       
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_loss = net_manager.get_reward(action, step, pre_loss)
            print("=====>", reward, pre_loss)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.storeRollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)

def main():
    global args
    args = parse_args()
    #out.csv is defined.
    data=pd.read_csv('out.csv')
    data.drop(data.filter(regex="Unname"),axis=1, inplace=True) 
    train(data)

if __name__ == '__main__':
  main()