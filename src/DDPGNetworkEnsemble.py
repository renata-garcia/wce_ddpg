from keras.layers import Dense, Concatenate

import tensorflow as tf
import numpy as np

import base.config as ddpg_cfg
import DDPGNetworkNode as node
from DDPGNetwork import DDPGNetwork
from DDPGNetworkNode import DDPGNetworkNode

from CriticAggregation import WeightedByTDError
from CriticAggregation import WeightedByTDErrorWeighing
from CriticAggregation import WeightedByTDErrorWeighingMax
from CriticAggregation import WeightedByTDErrorNormByExpV2
from CriticAggregation import WeightedByTDErrorNormByExp
from CriticAggregation import WeightedByTDErrorAndTailV2
from CriticAggregation import WeightedByTDErrorAndTail
from CriticAggregation import WeightedByTDErrorAndQ
from CriticAggregation import WeightedByTDErrorAnd05Q
from CriticAggregation import WeightedByTDErrorAnd01Q
from CriticAggregation import WeightedByTDErrorResetingWeightsUntil200
from CriticAggregation import WeightedByTDErrorResetingWeights
from CriticAggregation import WeightedByTDError1E3TarrgetEntropy
from CriticAggregation import WeightedByTDError1E4TarrgetEntropy
from CriticAggregation import WeightedByTDErrorLess1E3KEntropy
from CriticAggregation import WeightedByTDErrorLess1E4KEntropy
from CriticAggregation import WeightedByTDErrorAnd1E4KEntropy
from CriticAggregation import WeightedByAverage
from CriticAggregation import WeightedByFixedHalf
from CriticAggregation import WeightedByFixedOne
from CriticAggregation import WeightedByTDErrorAEntropy
from CriticAggregation import WeightedByTDErrorNorm001K

#rm DDPGNetworkEnsemble.py; touch DDPGNetworkEnsemble.py; chmod 755 DDPGNetworkEnsemble.py; nano DDPGNetworkEnsemble.py

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
        elif typeCriticAggregation == "TDErrorWeighing":
            q_critic = WeightedByTDErrorWeighing(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorWeighingMax":
            q_critic = WeightedByTDErrorWeighingMax(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorNormByExpV2":
            q_critic = WeightedByTDErrorNormByExpV2(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorNormByExp":
            q_critic = WeightedByTDErrorNormByExp(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAndTail":
            q_critic = WeightedByTDErrorAndTail(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAndTailV2":
            q_critic = WeightedByTDErrorAndTailV2(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAndQ":
            q_critic = WeightedByTDErrorAndQ(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAnd05Q":
            q_critic = WeightedByTDErrorAnd05Q(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAnd01Q":
            q_critic = WeightedByTDErrorAnd01Q(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorResetingWeightsUntil200":
            q_critic = WeightedByTDErrorResetingWeightsUntil200(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorResetingWeights":
            q_critic = WeightedByTDErrorResetingWeights(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError1E3TarrgetEntropy":
            q_critic = WeightedByTDError1E3TarrgetEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError1E4TarrgetEntropy":
            q_critic = WeightedByTDError1E4TarrgetEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorLess1E3KEntropy":
            q_critic = WeightedByTDErrorLess1E3KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorLess1E4KEntropy":
            q_critic = WeightedByTDErrorLess1E4KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorAnd1E4KEntropy":
            q_critic = WeightedByTDErrorAnd1E4KEntropy(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorNorm001EntropyV2":
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
            feed_dict[self._ensemble[j][0].a_in] = np.vstack(act)
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


