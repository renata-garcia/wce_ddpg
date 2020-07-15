from keras.layers import Dense, Concatenate
import tensorflow as tf
import base.config as ddpg_cfg

#rm DDPGNetwork.py; touch DDPGNetwork.py; chmod 755 DDPGNetwork.py; nano DDPGNetwork.py;
"""DDPG actor-critic network with two hidden layers"""


class DDPGNetwork(ddpg_cfg.DDPGNetworkConfig):
  def __init__(self, sess, obs, act, a_max, config):
    print(obs, act, a_max)
    self.session = sess
    self.layer1_size = config._layer1
    self.layer2_size = config._layer2
    self._lractor = config._lractor
    self._lrcritic = config._lrcritic
    self.act1 = config._act1
    self.act2 = config._act2

    prev_vars = len(tf.trainable_variables())

    # Actor network
    self.s_in = tf.placeholder(tf.float32, shape=(None, obs), name='s_in')
    ha1 = Dense(self.layer1_size, activation=self.act1[1:-1], name='h_actor1')(self.s_in)
    ha2 = Dense(self.layer2_size, activation=self.act1[1:-1], name='h_actor2')(ha1)
    self.a_out = a_max * Dense(act, activation='tanh', name='a_out')(ha2)
    theta = tf.trainable_variables()[prev_vars:]

    # Critic network
    self.a_in = tf.placeholder_with_default(tf.stop_gradient(self.a_out), shape=(None, act), name='a_in')
    hq1 = Dense(self.layer1_size, activation=self.act1[1:-1], name='h_critic1')(self.s_in)
    hc = Concatenate()([hq1, self.a_in])
    hq2 = Dense(self.layer2_size, activation=self.act1[1:-1], name='h_critic2')(hc)
    self.q = Dense(1, activation=self.act2[1:-1], name='q')(hq2)

    # Actor network update
    dq_da = tf.gradients(self.q, self.a_in, name='dq_da')[0]
    dq_dtheta = tf.gradients(self.a_out, theta, -dq_da, name='dq_dtheta')
    self.a_update = tf.train.AdamOptimizer(self._lractor).apply_gradients(zip(dq_dtheta, theta), name='a_update')

    # Critic network update
    self.q_target = tf.placeholder(tf.float32, shape=(None, 1), name='target')
    q_loss = tf.losses.mean_squared_error(self.q_target, self.q)
    self.q_update = tf.train.AdamOptimizer(self._lrcritic).minimize(q_loss, name='q_update')

  def get_value(self, obs, act=None):
    if act:
      return self.session.run(self.q, {self.s_in: obs, self.a_in: act})
    else:
      return self.session.run(self.q, {self.s_in: obs})

  def train(self, obs, act, q_target):
    # Train critic
    self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})

    # Train actor
    self.session.run(self.a_update, {self.s_in: obs})

