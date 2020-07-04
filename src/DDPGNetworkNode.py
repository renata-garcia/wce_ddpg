from keras.layers import Dense, Concatenate
import tensorflow as tf
import base.DDPGNetworkConfig as ddpg_cfg

#rm DDPGNetworkNode.py touch DDPGNetworkNode.py; chmod 755 DDPGNetworkNode.py; nano DDPGNetworkNode.py

class DDPGNetworkNode(ddpg_cfg.DDPGNetworkConfig):
  def __init__(self, sess, sin, qtarget, act, a_max, config):
    self.session = sess
    self.s_in = sin
    self.q_target = qtarget
    self.lr_actor = config._lractor
    self.lr_critic = config._lrcritic
    self.layer1_size = config._layer1
    self.layer2_size = config._layer2
    self.act1 = config._act1
    self.act2 = config._act2
    config.print()

    prev_vars = len(tf.trainable_variables())

    # Actor network
    ha1 = Dense(self.layer1_size, activation=self.act1[1:-1], name='h_actor1')(self.s_in)
    ha2 = Dense(self.layer2_size, activation=self.act1[1:-1], name='h_actor2')(ha1)
    self.a_out = a_max * Dense(act, activation='tanh', name='a_out')(ha2)
    theta = tf.trainable_variables()[prev_vars:]

    # Critic network
    self.a_in = tf.placeholder_with_default(tf.stop_gradient(self.a_out), shape=(None, act), name='a_in')
    hq1 = Dense(self.layer1_size, activation=self.act1[1:-1], name='h_critic1')(self.s_in)
    hc1 = Concatenate()([hq1, self.a_in])
    hq2 = Dense(self.layer2_size, activation=self.act1[1:-1], name='h_critic2')(hc1)
    self.q = Dense(1, activation=self.act2[1:-1], name='q1')(hq2)

    # Actor network update
    dq_da = tf.gradients(self.q, self.a_in, name='dq_da')[0]
    dq_dtheta = tf.gradients(self.a_out, theta, -dq_da, name='dq_dtheta')
    self.a_update = tf.train.AdamOptimizer(self.lr_actor).apply_gradients(zip(dq_dtheta, theta), name='a_update')

    # Critic network update
    q_loss = tf.losses.mean_squared_error(self.q_target, self.q)
    self.q_update = tf.train.AdamOptimizer(self.lr_critic).minimize(q_loss, name='q_update')

  def get_value(self, obs):
    return self.session.run(self.q, {self.s_in: obs})

  def train(self, obs, act, q_target):
    # Train critic and Train actor
    self.session.run([self.q_update,self.a_update], {self.s_in: obs, self.a_in: act, self.q_target: q_target})

