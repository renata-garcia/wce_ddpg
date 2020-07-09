from keras.layers import Dense, Concatenate
import tensorflow as tf
import base.DDPGNetworkConfig as ddpg_cfg
import DDPGNetworkNode as node

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


  def get_value(self, main_target, obs):
    returns = []
    for ii in range(self._num_ensemble):
      returns.append(self._ensemble[ii][main_target].q)

    return self._session.run(returns, {self._sin: obs})


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


