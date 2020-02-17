from keras.layers import Dense, Concatenate
import tensorflow as tf

import os

class WeightCritic():
  def __init__(self, sess, qin, td, lrcritic):
    self.session = sess
    self.q_in = qin
    self.td_ = td
    self.lr_critic = lrcritic

    q_critic_d = Dense(1, name='q_critic_d')(self.q_in)
    self.weights_t = tf.get_default_graph().get_tensor_by_name(os.path.split(q_critic_d.name)[0] + '/kernel:0') + 0.001
    self.weights_raw = tf.transpose(self.weights_t, name='self.weights_t')
    self.weights = tf.nn.softmax(self.weights_raw)
    self.q_critic = tf.reduce_max((self.q_in * self.weights) / tf.reduce_sum(self.weights, 1))

    qs_loss = tf.reduce_sum(((self.td_ ** 2) * self.weights) / tf.reduce_sum(self.weights))
    self.qs_update = tf.train.AdamOptimizer(self.lr_critic).minimize(qs_loss, name='qs_update')

  def train(self, qsin, td):
    self.session.run(self.qs_update, {self.q_in: qsin, self.td_: td})

