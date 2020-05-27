import tensorflow as tf
import keras
import numpy as np
import os

class WeightedByTDErrorAddingReward:

    def __init__(self, sess, qin, td, num_ensemble):
        # super().__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')
        #q_critic_d = Dense(1, name='q_critic_d')(self._q_in)
        #self.weights_t = tf.get_default_graph().get_tensor_by_name(os.path.split(q_critic_d.name)[0] + '/kernel:0') + 0.001

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        # self.weights_raw = tf.transpose(self.weights_t, name='self.weights_t')
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        # TODO testar retirando o treinamento, resultado deve ser igual a ter média
        qs_loss = tf.reduce_sum(((self._td ** 2) * self.weights))
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]
