import types
from keras.layers import Dense, Concatenate
import tensorflow as tf
import numpy as np

#rm CriticAggregation.py; touch CriticAggregation.py; chmod 755 CriticAggregation.py; nano CriticAggregation.py
class CriticAggregation(object):
    def __init__(self, func=None):
        if func is not None:
            # take a function, bind it to this instance, and replace the default bound method 'execute' with this new bound method.
            self.init = types.MethodType(func, self)
            if ('{}'.format(func.__name__))== "weighted_by_TD_error":
                self.train = types.MethodType(train_weighted_by_TD_error, self)
            else:
                self.train = types.MethodType(init, self) #TODO check again
            self.name = '{}_{}'.format(self.__class__.__name__, func.__name__)
        else:
            self.name = '{}_default'.format(self.__class__.__name__)

    def init(self):
        print('Default method')
        print('{}\n'.format(self.name))

    def weighted_by_TD_error(self, sess, qin, td, num_ensemble):
        print('Replacement1 method')
        print('{}\n'.format(self.name))

        self.session = sess
        self.q_in = qin
        self.td_ = td
        self.lr_critic = 0.0001

        q_critic_d = Dense(1, name='q_critic_d')(self.q_in)
        # self.weights_t = tf.get_default_graph().get_tensor_by_name(os.path.split(q_critic_d.name)[0] + '/kernel:0') + 0.001
        self.weights_raw = tf.get_variable(name='asdsas', dtype=tf.float32,
                                           initializer=np.zeros(num_ensemble, np.float32))
        # self.weights_raw = tf.transpose(self.weights_t, name='self.weights_t')
        self.weights = tf.nn.softmax(self.weights_raw)
        self.q_critic = tf.reduce_sum((self.q_in * self.weights))

        # TODO testar retirando o treinamento, resultado deve ser igual a ter média
        self.qs_loss_int = ((self.td_ ** 2) * self.weights)  # TODO verificar size/len
        self.qs_loss_int = (self.weights)  # TODO verificar size/len
        self.qs_loss = tf.reduce_sum(((self.td_ ** 2) * self.weights))  # TODO verificar size/len
        self.qs_update = tf.train.AdamOptimizer(self.lr_critic).minimize(self.qs_loss, name='qs_update')

    def train_weighted_by_TD_error(self, td):  # TODO qsin no need, remove
        self.session.run([self.qs_update, self.qs_loss_int], {self.q_in: qsin, self.td_: td})
        # print(r_ql.shape)
        # print(r_ql)

    def weighted_by_average(self, sess, qin, td, num_ensemble):
        print('Replacement2 method')
        print('{}\n'.format(self.name))
