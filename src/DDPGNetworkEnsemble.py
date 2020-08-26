from keras.layers import Dense, Concatenate

import tensorflow as tf
import numpy as np

import base.config as ddpg_cfg
import DDPGNetworkNode as node
from DDPGNetwork import DDPGNetwork
from DDPGNetworkNode import DDPGNetworkNode

from CriticAggregation import WeightedByTDError
from CriticAggregation import WeightedByTDErrorNormValue
from CriticAggregation import WeightedByTDError1KEntropy
from CriticAggregation import WeightedByTDError10KEntropy
from CriticAggregation import WeightedByTDError100KEntropy
from CriticAggregation import WeightedByTDError1000KEntropy
from CriticAggregation import WeightedByAverage
from CriticAggregation import WeightedByFixedHalf
from CriticAggregation import WeightedByFixedOne
from CriticAggregation import WeightedByTDErrorAEntropy
from CriticAggregation import WeightedByTDErrorNorm001K

#rm DDPGNetworkNode.py touch DDPGNetworkNode.py; chmod 755 DDPGNetworkNode.py; nano DDPGNetworkNode.py

class DDPGNetworkEnsemble(ddpg_cfg.DDPGNetworkConfig):
    def __init__(self, sess, sin, cfg_ens, action_space, num_ensemble, max_action):
        self._session = sess
        self._sin = sin
        self._num_ensemble = num_ensemble
        self._ensemble = []
        self._qtarget = []
        for ii in range(self._num_ensemble):
            self._qtarget.append(tf.placeholder(tf.float32, shape=(None, 1), name='target'))

        for ne in range(num_ensemble):
            prev_vars = len(tf.trainable_variables())
            network = node.DDPGNetworkNode(self._session, sin, self._qtarget[ne], action_space, max_action,
                                                      cfg_ens[ne]['config_ddpg'])
            target_network = node.DDPGNetworkNode(self._session, sin, self._qtarget[ne], action_space, max_action,
                                                  cfg_ens[ne]['config_ddpg'])
            vars = tf.trainable_variables()[prev_vars:]
            tau = cfg_ens[ne]['tau']
            update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                          ix, var in enumerate(vars[0:len(vars) // 2])]
            self._ensemble.append((network, target_network, update_ops))


    def choose_aggregation(self, typeCriticAggregation, session, qs1, qin, td):
        if typeCriticAggregation == "Average":
            q_critic = WeightedByAverage(session, qs1, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError":
            q_critic = WeightedByTDError(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorNormValue":
            q_critic = WeightedByTDErrorNormValue(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError1KEntropy":
            q_critic = WeightedByTDError1KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError10KEntropy":
            q_critic = WeightedByTDError10KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError100KEntropy":
            q_critic = WeightedByTDError100KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError1000KEntropy":
            q_critic = WeightedByTDError1000KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorNorm001Entropy":
            q_critic = WeightedByTDErrorNorm001K(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAEntropy":
            q_critic = WeightedByTDErrorAEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "FixedByHalf":
            q_critic = WeightedByFixedHalf(session, qin, self._num_ensemble)
        elif typeCriticAggregation == "FixedOneZero":
            q_critic = WeightedByFixedOne(session, qin, self._num_ensemble, 0)
        elif typeCriticAggregation == "FixedOneOne":
            q_critic = WeightedByFixedOne(session, qin, self._num_ensemble, 1)
        elif typeCriticAggregation == "FixedOneTwo":
            q_critic = WeightedByFixedOne(session, qin, self._num_ensemble, 2)
        else:
            print("typeCriticAggregation")
            print(typeCriticAggregation)
            exit(-1)
        return q_critic

    #TODO actions = get_all_actions_target_network(nobs)
    #TODO nextq = max(ensemble_q_values_target_network(nobs, actions))
    #TODO target = r + gamma * nextq

    #TODO Q = Q(s, a)
    #TODO Q_Target_Treino = r + gamma *max_{a' \in ensemble_actions}Q(s', a')
    #TODO Q_Target_TD_error = r + gamma *max_{a' \in ensemble_actions}Q(s', a')
    #TODO TD_error = Q_target - Q

    #TODO Q = Q(s, a)
    #TODO Q_Target = r + gamma *max_{a' \in ensemble_actions}Q(s', a')
    #TODO TD_error = Q_target - Q

    #TODO train_nextq_results = ddpgne.get_value(1, nobs)  # TODO use action ensemble
    def get_value(self, main_target, obs, act=None):
        returns = []
        for ii in range(self._num_ensemble):
            returns.append(self._ensemble[ii][main_target].q)

        if not act is None:
            feed_dict = {self._sin: obs}
            for ii in range(self._num_ensemble):
                feed_dict[self._ensemble[ii][main_target].a_in] = act
            return self._session.run(returns, feed_dict)
        else:
            return self._session.run(returns, {self._sin: obs})


    def build_value_function(self, typeCriticAggregation):
        qs1 = []
        for i in range(self._num_ensemble):
            qs1.append(self._ensemble[i][0].q)
        qin = tf.placeholder_with_default(tf.stop_gradient(qs1), shape=(None, self._num_ensemble, 1), name='qin')
        td = tf.placeholder(tf.float32, shape=(None, self._num_ensemble), name='td')
        q_critic = self.choose_aggregation(typeCriticAggregation, self._session, qs1, qin, td)
        q_critic.buildLayer()
        return q_critic


    def train(self, obs, act, q_target):
        # self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})
        # self.session.run(self.a_update, {self.s_in: obs})

        # Train critic
        feed_dict = {self._sin :  obs}
        for j in range(self._num_ensemble):
            feed_dict[self._ensemble[j][0].a_in] = act
            feed_dict[self._ensemble[j][0].q_target] = q_target[j]

        returns = []
        for ii in range(self._num_ensemble):
           returns.append(self._ensemble[ii][0].q_update)

        self._session.run(returns, feed_dict)

        # Train actor
        returns = []
        for ii in range(self._num_ensemble):
            returns.append(self._ensemble[ii][0].a_update)

        feed_dict = {self._sin: obs}
        self._session.run(returns, feed_dict)


class DDPGNetworkSingle(ddpg_cfg.DDPGNetworkConfig):
    #TODO not finished
    def __init__(self, sess, sin, cfg_ens, action_space, num_ensemble, max_action, hasTargetActionInfo):
        self._session = sess
        self._sin = sin
        self._num_ensemble = num_ensemble
        self._ensemble = []
        self._qtarget = []
        for ii in range(self._num_ensemble):
            self._qtarget.append(tf.placeholder(tf.float32, shape=(None, 1), name='target'))

        for ne in range(num_ensemble):
            prev_vars = len(tf.trainable_variables())
            network = DDPGNetworkNode(self._session, sin, self._qtarget[ne], action_space, max_action,
                                      cfg_ens[ne]['config_ddpg'])
            target_network = DDPGNetworkNode(self._session, sin, self._qtarget[ne], action_space, max_action,
                                             cfg_ens[ne]['config_ddpg'])
            vars = tf.trainable_variables()[prev_vars:]
            tau = cfg_ens[ne]['tau']
            update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value()))
                          for ix, var in enumerate(vars[0:len(vars) // 2])]
            self._ensemble.append((network, target_network, update_ops))


    def get_value(self, main_target, obs, act=None):
        if act:
            return self._session.run(self._ensemble[0][main_target].q, {self._sin: obs, self._ensemble[0][main_target].a_in: act})
        else:
            return self._session.run(self._ensemble[0][main_target].q, {self._sin: obs})


    def train(self, obs, act, q_target):
        # self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})
        # self.session.run(self.a_update, {self.s_in: obs})

        # Train critic
        feed_dict = {self._sin :  obs}
        for j in range(self._num_ensemble):
            feed_dict[self._ensemble[j][0].a_in] = act
            feed_dict[self._ensemble[j][0].q_target] = q_target[j]

        returns = []
        for ii in range(self._num_ensemble):
           returns.append(self._ensemble[ii][0].q_update)

        self._session.run(returns, feed_dict)


        # Train actor
        feed_dict = {self._sin: obs}

        returns = []
        for ii in range(self._num_ensemble):
            returns.append(self._ensemble[ii][0].a_update)

        self._session.run(returns, feed_dict)


