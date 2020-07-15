from keras.layers import Dense, Concatenate
import tensorflow as tf
import base.config as ddpg_cfg
import DDPGNetworkNode as node

from CriticAggregation import WeightedByTDError
from CriticAggregation import WeightedByAverage
from CriticAggregation import WeightedByTDErrorInvW
from CriticAggregation import WeightedByTDErrorAddingReward #TODO
from CriticAggregation import WeightedByTDErrorAndReward #TODO
from CriticAggregation import WeightedByAddingReward #TODO
from CriticAggregation import WeightedByFixedHalf
from CriticAggregation import WeightedByFixedOne

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
            target_network = node.DDPGNetworkNode(self._session, sin, self._qtarget[ne], action_space,
                                                       max_action, cfg_ens[ne]['config_ddpg'])
            vars = tf.trainable_variables()[prev_vars:]
            tau = cfg_ens[ne]['tau']
            update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                          ix, var in enumerate(vars[0:len(vars) // 2])]
            self._ensemble.append((network, target_network, update_ops))


    def choose_aggregation(self, typeCriticAggregation, session, qs1, qin, td):
        if typeCriticAggregation == "Average":
            q_critic = WeightedByAverage(session, qs1, td, self._num_ensemble)
        elif typeCriticAggregation == "TDErrorInvW":
            q_critic = WeightedByTDErrorInvW(session, qin, td, self._num_ensemble)
        elif typeCriticAggregation == "TDError":
            q_critic = WeightedByTDError(session, qin, td, self._num_ensemble)
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


    def get_value(self, main_target, obs):
        returns = []
        for ii in range(self._num_ensemble):
           returns.append(self._ensemble[ii][main_target].q)

        return self._session.run(returns, {self._sin: obs})


    def get_value_function(self, typeCriticAggregation):
        qs1 = []
        for i in range(self._num_ensemble):
            qs1.append(self._ensemble[i][0].q)
        qin = tf.placeholder_with_default(tf.stop_gradient(tf.reshape(qs1, [1, self._num_ensemble])), shape=(None, self._num_ensemble), name='qin')
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
            # print(act)
            # print(act[0])
            # print(len(act))
            feed_dict[self._ensemble[j][0].q_target] = q_target[j]
            # print(len(q_target[j]))

        returns = []
        for ii in range(self._num_ensemble):
           returns.append(self._ensemble[ii][0].q_update)

        # print("##########")
        # print(returns)
        # print(len(returns))
        # print(feed_dict)
        # print(len(feed_dict))
        self._session.run(returns, feed_dict)


        # Train actor
        feed_dict = {self._sin: obs}

        returns = []
        for ii in range(self._num_ensemble):
            returns.append(self._ensemble[ii][0].a_update)

        self._session.run(returns, feed_dict)



class DDPGNetworkSingle(ddpg_cfg.DDPGNetworkConfig):
    #TODO not finished
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
            network = DDPGNetwork(session, env._env.observation_space.shape[0]+1, action_space, max_action, cfg_ens[0]['config_ddpg'])
            target_network = DDPGNetwork(session, env._env.observation_space.shape[0]+1, action_space, max_action, cfg_ens[0]['config_ddpg'])
            vars = tf.trainable_variables()[prev_vars:]
            tau = cfg_ens[ne]['tau']
            update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value()))
                          for ix, var in enumerate(vars[0:len(vars) // 2])]
            self._ensemble.append((network, target_network, update_ops))


    def get_value(self, main_target, obs):
        returns = []
        # for ii in range(self._num_ensemble):
        #    returns.append(self._ensemble[ii][main_target].q)
        #
        # return self._session.run(returns, {self._sin: obs})


    def get_value_function(self):
        return getattr(network, "q")


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


