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


class WeightedByTDErrorAndTail(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAndTail, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorAndTail_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))
        td2td = (self._td ** 2)
        qs_loss = tf.reduce_sum( td2td * self.weights + 2*tf.exp(-0.1*td2td + 1e-10) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorAndQ(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAndQ, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self.td_error = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum( ((self._q_in** 2) * self.weights) )

        self.td_error = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_loss = self.td_error + self.q_critic
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorAnd05Q(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAnd05Q, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self.td_error = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum( ((self._q_in** 2) * self.weights) )

        self.td_error = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_loss = self.td_error + self.q_critic
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorAnd01Q(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAnd01Q, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self.td_error = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum( ((self._q_in** 2) * self.weights) )

        self.td_error = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_loss = self.td_error + self.q_critic
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorResetingWeights(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorResetingWeights, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self._reseted = 0
        self._n_reseted = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        self._weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(self._weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self._qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self._qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        # if (ep % 50 > 0) and not self._reseted:
        if ep/50 > self._n_reseted:
            self._n_reseted = self._n_reseted + 1;
            wr = self._session.run(self._weights_raw)
            w = self._session.run(self.weights)
            qslss = self._session.run(self._qs_loss, {self._td:td})
            print(ep)
            print(wr)
            print(w)
            print(qslss)
            self._session.run(self._weights_raw.initializer)
            wr = self._session.run(self._weights_raw)
            print(wr)
            print("RESETING****************************************************");
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorPrintingQloss(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorResetingWeights, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self._n_reseted = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        self._weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(self._weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self._qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self._qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        if ep/50 > self._n_reseted:
            self._n_reseted = self._n_reseted + 1;
            wr = self._session.run(self._weights_raw)
            w = self._session.run(self.weights)
            qslss = self._session.run(self._qs_loss, {self._td:td})
            print(ep)
            print(wr)
            print(w)
            print(qslss)
            print("RESETING****************************************************");
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorResetingWeightsUntil200(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorResetingWeightsUntil200, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self._reseted = 0
        self._n_reseted = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDError_buildLayer')

        self._weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(self._weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self._qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self._qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        if (ep/50 > self._n_reseted) and not self._reseted:
            if ep > 200:
                self._reseted = 1
            self._n_reseted = self._n_reseted + 1
            self._session.run(self._weights_raw.initializer)
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorResetingWeights(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorResetingWeights, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0
        self._n_reseted = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorResetingWeights_buildLayer')

        self._weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(self._weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) )
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        if ep/50 > self._n_reseted:
            self._n_reseted = self._n_reseted + 1
            self._session.run(self._weights_raw.initializer)
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorLess1E3KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorLess1E3KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorLess1E3KEntropy_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self.qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights)) - 1e3*tf.reduce_sum(-self.weights*tf.math.log(1e-10+self.weights)) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorLess1E4KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorLess1E4KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorLess1E4KEntropy_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self.qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights)) - 1e4*tf.reduce_sum(-self.weights*tf.math.log(1e-10+self.weights)) #-entropy (trying to maximize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorAnd1E4KEntropy(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorAnd1E4KEntropy, self).__init__()
        self._session = sess
        self._q_in = qin
        self._td = td
        self._lr_critic = 0.0001
        self._num_ensemble = num_ensemble
        self.q_critic = 0

    def buildLayer(self):
        print('CriticAggregation::WeightedByTDErrorAnd1E4KEntropy_buildLayer')

        weights_raw = tf.get_variable(name='weights_raw', dtype=tf.float32,
                                           initializer=np.zeros(self._num_ensemble, np.float32))
        self.weights = tf.nn.softmax(weights_raw)
        self.q_critic = tf.reduce_sum((self._q_in * self.weights))

        self.qs_loss = tf.reduce_sum( ((self._td ** 2) * self.weights) ) + 1e4*tf.reduce_sum(-self.weights*tf.math.log(1e-10+self.weights)) #+entropy (trying to minimize entropy)
        self.qs_update = tf.train.AdamOptimizer(self._lr_critic).minimize(self.qs_loss, name='qs_update')

    def train(self, td, addrw, ep):
        return self._session.run([self.qs_update, self.weights], {self._td: td})[1]


class WeightedByTDErrorNorm001K(CriticAggregation):

    def __init__(self, sess, qin, td, num_ensemble):
        super(WeightedByTDErrorNorm001K, self).__init__()
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
        qs_loss = tf.reduce_sum( ((self._td ** 2)/td2td_max * self.weights)  - 0.01*(-self.weights*tf.math.log(1e-10+self.weights)) )
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
        qs_loss = tf.reduce_sum( ((self._td ** 2)/td2td_max * self.weights) - K*(-self.weights*tf.math.log(1e-10+self.weights)) ) #-entropy (trying to maximize entropy)
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
        fixed = 1 / self.num_ensemble_
        self.weights = [fixed for ii in range(self.num_ensemble_)]

    def buildLayer(self):
        print('CriticAggregation::WeightedByAverage_buildLayer')
        self.q_critic = keras.layers.average(self._q_in)

    def train(self, td, addrw, ep):
        return self.weights

