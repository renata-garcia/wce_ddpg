from keras.layers import Dense, Concatenate
import tensorflow as tf

class DDPGNetworkNode():
  def __init__(self, sess, sin, qtarget, act, a_max, lractor, lrcritic):
    self.session = sess
    self.s_in = sin
    self.q_target = qtarget

    self.layer1_size = 400
    self.layer2_size = 300
    self.lr_actor = lractor
    self.lr_critic = lrcritic

    prev_vars = len(tf.trainable_variables())

    # Actor network
    ha1 = Dense(self.layer1_size, activation='relu', name='h_actor1')(self.s_in)
    ha2 = Dense(self.layer2_size, activation='relu', name='h_actor2')(ha1)
    self.a_out = a_max * Dense(act, activation='tanh', name='a_out')(ha2)
    theta = tf.trainable_variables()[prev_vars:]

    # Critic network
    self.a_in = tf.placeholder_with_default(tf.stop_gradient(self.a_out), shape=(None, act), name='a_in')
    hq1 = Dense(self.layer1_size, activation='relu', name='h_critic1')(self.s_in)
    hc1 = Concatenate()([hq1, self.a_in])
    hq2 = Dense(self.layer2_size, activation='relu', name='h_critic2')(hc1)
    self.q = Dense(1, activation='linear', name='q1')(hq2)

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
    # Train critic
    self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})
    # Train actor
    self.session.run(self.a_update, {self.s_in: obs})

