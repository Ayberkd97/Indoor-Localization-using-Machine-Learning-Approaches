import tensorflow as tf
import random
import numpy as np

class Reinforce():
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 division_rate=100.0,
                 discount_factor=0.99,
                 exploration=0.3): #The probability of generating random action is exploration
        #TensorFlow session and optimizer, will be initialized separately.
        self.sess = sess
        self.optimizer = optimizer 
        self.policy_network = policy_network #parameter is defined in RNN layer training network in policy network function
        self.division_rate = division_rate #Normal distribution values of each neuron from -1.0 to 1.0.
        self.discount_factor=discount_factor
        self.max_layers = max_layers #number of maximum layers structure can reach. Manually defined.
        self.global_step = global_step
        self.exploration = exploration

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))#A class for running TensorFlow operations.

    def get_action(self, state):
        if random.random() < self.exploration:
            return np.array([[random.sample(range(1, 100), 2*self.max_layers)]])
        else:
            return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(tf.float32, [None, self.max_layers*1], name="states")#Inserts a placeholder for a tensor that will be always fed.

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):#A context manager for defining ops that creates variables (layers).
                self.policy_outputs = self.policy_network(self.states, self.max_layers)#policy network function

            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")#Return a Tensor with the same shape and contents as input.

            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name="predicted_action")
            #Casts a tensor to a new type.

        # regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network(self.states, self.max_layers)
                print("self.logprobs", self.logprobs)

            # compute policy loss and regularization loss
            self.difference_loss = tf.squared_difference(self.logprobs[:, -1, :], self.states)
            #Computes softmax cross entropy between logits and labels. Measures the probability error in discrete classification tasks in which the classes are mutually exclusive
            self.pg_loss            = tf.reduce_mean(self.difference_loss) #Computes the mean of elements across dimensions of a tensor.
            self.loss               = self.pg_loss

            #compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)
            
            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)
                #Apply gradients to variables.This is the second part of minimize(). It returns an Operation that applies gradients.

    #These are training steps
    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        rewars = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss],
                     {self.states: states,
                      self.discounted_rewards: rewars})
        return ls