import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, max_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
epsilon = 1e-6

class Siamese:
    def __init__(self):
        self.input_1 = tf.placeholder(tf.float32, [None, 10], name='input_1')
        self.input_2 = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.input_1)
            scope.reuse_variables()
            self.o2 = self.network(self.input_2)
        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.contrastive_loss()

    def network(self, x):
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        b = tf.get_variable(name+'b', dtype=tf.float32, shape=[n_weight], initializer=tf.zeros_initializer())
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def contrastive_loss(self):
        margin = 0.01
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi") 

        C = tf.constant(margin, name="C")
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + epsilon, name="eucd")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")

        
        return loss
