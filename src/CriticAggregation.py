import tensorflow as tf
import keras
import numpy as np
import os

#rm CriticAggregation.py; touch CriticAggregation.py; chmod 755 CriticAggregation.py; nano CriticAggregation.py

class CriticAggregation(object):
    def __init__(self):
        print("class CriticAggregation(object)")

class WeightedByTDError(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDError1KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError1KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) - 1*(-self.weights*tf.math.log(self.weights)) ) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDError10KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError10KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) -10*(-self.weights*tf.math.log(self.weights)) ) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDError100KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError100KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) -100*(-self.weights*tf.math.log(self.weights)) ) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDError1000KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError1000KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) -1000*(-self.weights*tf.math.log(self.weights)) ) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDErrorNorm001K(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDError, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        td2td_max = tf.reduce_max((self._td ** 2))
        qs_loss = tf.reduce_sum( ((self._td ** 2)/td2td_max * self.weights)  - 0.01*(-self.weights*tf.math.log(self.weights)) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDErrorAEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self.k_entropy = 10

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        td2td_max = tf.reduce_max((self._td ** 2))
        K = 10/(100*(self.k_entropy//10))
        qs_loss = tf.reduce_sum( ((self._td ** 2)/td2td_max * self.weights) - K*(-self.weights*tf.math.log(self.weights)) ) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        self.k_entropy = ep
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDErrorInvW(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorInvW, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorInvW_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        # self.weights_raw = tf.transpose(self.weights_t, name='self.weights_t')
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * (1-self.weights)))

        # TODO testar retirando o treinamento, resultado deve ser igual a ter média
        qs_loss = tf.reduce_sum(((self._td ** 2) * self.weights))
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]

class WeightedByTDErrorAddingReward(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble, adding_reward):
        super().__init__()
        self._session = sess
        self._q_in = qin
        self._adding_reward_in = adding_reward
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorAddingReward_buildLayer')

        self._adding_reward = tf.constant(np.zeros(self._num_ensemble, np.float32))
        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))

        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * (self.weights + self._adding_reward_in)))

        # TODO testar retirando o treinamento, resultado deve ser igual a ter média
        qs_loss = tf.reduce_sum(((self._td ** 2) * self.weights))
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        # return self._session.run([self.qs_update, self.weights], {self._td: td})[1]
        qs_update_t, weights_t,  adding_reward_t,  adding_reward_in_t = self._session.run([self.qs_update, self.weights,  self._adding_reward, self._adding_reward_in], {self._td: td, self._adding_reward_in: addrw})
        return weights_t

class WeightedByTDErrorAndReward(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble, adding_reward):
        super().__init__()
        self._session = sess
        self._q_in = qin
        self._adding_reward_in = adding_reward
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorAndReward_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw) + self._adding_reward_in
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum(((self._td ** 2) * self.weights))
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return  self._session.run([self.qs_update, self.weights,  self._adding_reward_in], {self._td: td, self._adding_reward_in: addrw})[1]

class WeightedByAddingReward(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble, adding_reward):
        super().__init__()
        self._session = sess
        self._q_in = qin
        self._adding_reward_in = adding_reward
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorAndReward_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self._adding_reward_in))

        qs_loss = tf.reduce_sum(((self._td ** 2) * self.weights * self._adding_reward_in))
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        self._session.run([self.qs_update ], {self._td: td, self._adding_reward_in: addrw})
        return addrw

class WeightedByFixedHalf(CriticAggregation):

    def __init__(self, sess, qin, num_ensemble):
        super().__init__()
        self._session = sess
        self._q_in = qin
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self.fixed = tf.constant((1/self._num_ensemble) * np.ones(self._num_ensemble, np.float32))


    def buildLayer(self):
        self.q_critic = tf.reduce_sum((self._q_in * self.fixed))

    def train(self, td, addrw, ep):
        return  (1/self._num_ensemble) * np.ones(self._num_ensemble, np.float32)

class WeightedByFixedOne(CriticAggregation):

    def __init__(self, sess, qin, num_ensemble, fixed_one):
        super().__init__()
        self._session = sess
        self._q_in = qin
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self._fixed_one = fixed_one
        self._fixed = np.zeros(self._num_ensemble, np.float32)
        self._fixed[self._fixed_one] = 1.0
        self.fixed = tf.constant(self._fixed)

    def buildLayer(self):
        print('CriticAggregation::WeightedByFixedOne')

        self.q_critic = tf.reduce_sum((self._q_in * self.fixed))

    def train(self, td, addrw, ep):
        r = np.zeros(self._num_ensemble, np.float32)
        r[self._fixed_one] = 1.0
        return  r

class WeightedByAverage(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByAverage, self).__init__()
        self._session = sess
        self._q_in = qin
        self.q_critic = 0
        self.num_ensemble_ = num_ensemble

    def buildLayer(self):
        print('CriticAggregation::WeightedByAverage_buildLayer')
        self.q_critic = keras.layers.average(self._q_in)

    def train(self, td, addrw, ep):
        fixed = 1/self.num_ensemble_
        run_weights = [fixed for ii in range(self.num_ensemble_)]
        return run_weights
