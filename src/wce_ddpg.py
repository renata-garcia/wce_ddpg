#!/usr/bin/python3
#
# OpenAI Gym example using GRL environment
# This example must be run from its own directory, with
# grl installed to a path in LD_LIBRARY_PATH and grlpy installed
# to a path in PYTHON_PATH.
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import math, time, numpy as np
import gym
from gym.envs.registration import register

# TODO: checar esses imports
# Imports
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
import tensorflow as tf
from keras.layers import Dense, Concatenate, average
import random
import sys
from scipy import stats
import scipy

# Register environmnent instantiation. Every configuration file
# requires a different instantiation, as Gym does not allow passing
# parameters to an environment.
# The configuration must define an "environment" tag at the root that
# specifies the environment to be used.
print(sys.argv)
print(len(sys.argv))

if len(sys.argv) == 1:
  enable_ensemble = 1
  lr_actor = 0.001
  lr_critic = 0.0001
  replay_steps = 64
  size_batch = 16
  gamma = 0.99
  tau = 0.01
  episodes = 4000
  num_ensemble = 3
elif len(sys.argv) != 9:
  print(sys.argv)
  sys.exit(1)
else:
  enable_ensemble = int(sys.argv[1]) #1;
  lr_actor = float(sys.argv[2]) #0.0001
  lr_critic = float(sys.argv[3]) #0.001
  replay_steps = int(sys.argv[4]) #64
  size_batch = int(sys.argv[5]) #16
  gamma = float(sys.argv[6]) #0.99
  tau = float(sys.argv[7]) #0.001
  episodes = int(sys.argv[8]) #1000

print("simple_ddpg: ", enable_ensemble, "lr_actor: ", lr_actor, ", lr_critic: ", lr_critic, "\n",
      ", replay_steps: ", replay_steps, ", size_batch: ", size_batch, ", gamma: ", gamma, "\n",
      ", tau: ", tau, ", episodes: ", episodes)

# DDPG helper functions

"""DDPG actor-critic network with two hidden layers"""

class DDPGNetwork():
  def __init__(self, sess, obs, act, a_max):
    print(obs, act, a_max)
    self.session = sess
    self.layer1_size = 400
    self.layer2_size = 300

    prev_vars = len(tf.trainable_variables())

    # Actor network
    self.s_in = tf.placeholder(tf.float32, shape=(None, obs), name='s_in')
    ha1 = Dense(self.layer1_size, activation='relu', name='h_actor1')(self.s_in)
    ha2 = Dense(self.layer2_size, activation='relu', name='h_actor2')(ha1)
    self.a_out = a_max * Dense(act, activation='tanh', name='a_out')(ha2)
    theta = tf.trainable_variables()[prev_vars:]

    # Critic network
    self.a_in = tf.placeholder_with_default(tf.stop_gradient(self.a_out), shape=(None, act), name='a_in')
    hq1 = Dense(self.layer1_size, activation='relu', name='h_critic1')(self.s_in)
    hc = Concatenate()([hq1, self.a_in])
    hq2 = Dense(self.layer2_size, activation='relu', name='h_critic2')(hc)
    self.q = Dense(1, activation='linear', name='q')(hq2)

    # Actor network update
    dq_da = tf.gradients(self.q, self.a_in, name='dq_da')[0]
    dq_dtheta = tf.gradients(self.a_out, theta, -dq_da, name='dq_dtheta')
    self.a_update = tf.train.AdamOptimizer(lr_actor).apply_gradients(zip(dq_dtheta, theta), name='a_update')

    # Critic network update
    self.q_target = tf.placeholder(tf.float32, shape=(None, 1), name='target')
    q_loss = tf.losses.mean_squared_error(self.q_target, self.q)
    self.q_update = tf.train.AdamOptimizer(lr_critic).minimize(q_loss, name='q_update')

  def get_value(self, obs, act=None):
    if act:
      return self.session.run(self.q, {self.s_in: obs, self.a_in: act})
    else:
      return self.session.run(self.q, {self.s_in: obs})

  def train(self, obs, act, q_target, noise=None):
    # Train critic
    self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})

    # Train actor
    self.session.run(self.a_update, {self.s_in: obs})

# DDPG helper functions

"""DDPG actor-critic network with two hidden layers"""

class DDPGNetworkNode():
  def __init__(self, sess, sin, qtarget, act, a_max):
    self.session = sess
    self.s_in = sin
    self.q_target = qtarget

    self.layer1_size = 400
    self.layer2_size = 300

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
    self.a_update = tf.train.AdamOptimizer(lr_actor).apply_gradients(zip(dq_dtheta, theta), name='a_update')

    # Critic network update
    q_loss = tf.losses.mean_squared_error(self.q_target, self.q)
    self.q_update = tf.train.AdamOptimizer(lr_critic).minimize(q_loss, name='q_update')

  def get_value(self, obs):
    return self.session.run(self.q, {self.s_in: obs})

  def train(self, obs, act, q_target):
    # Train critic
    self.session.run(self.q_update, {self.s_in: obs, self.a_in: act, self.q_target: q_target})
    # Train actor
    self.session.run(self.a_update, {self.s_in: obs})


class WeightCritic():
  def __init__(self, sess, qin, td):
    self.session = sess
    self.q_in = qin
    self.td_ = td

    q_critic_d = Dense(1, name='q_critic_d')(self.q_in)
    self.weights_t = tf.get_default_graph().get_tensor_by_name(os.path.split(q_critic_d.name)[0] + '/kernel:0') + 0.0001
    self.weights_raw = tf.transpose(self.weights_t, name='self.weights_t')
    self.weights = tf.nn.softmax(self.weights_raw)
    self.q_critic = tf.reduce_max((self.q_in * self.weights) / tf.reduce_sum(self.weights, 1))
    print("q_critic_d")
    print(q_critic_d)
    print("self.weights")
    print(self.weights)
    print("self.q_critic")
    print(self.q_critic)

    qs_loss = tf.reduce_sum(((self.td_ ** 2) * self.weights) / tf.reduce_sum(self.weights))
    print(qs_loss)
    self.qs_update = tf.train.AdamOptimizer(lr_critic).minimize(qs_loss, name='qs_update')

  def train(self, qsin, td):
    self.session.run(self.qs_update, {self.q_in: qsin, self.td_: td})

"""Memory of all past transitions for experience replay"""

class ReplayMemory():
  def __init__(self):
    self.memory = []

  def sample_minibatch(self, batch_size):
    batch = [self.memory[random.randint(0, len(self.memory) - 1)] for a in range(batch_size)]
    obs = [e[0] for e in batch]
    act = [e[1] for e in batch]
    rew = [e[2] for e in batch]
    nobs = [e[3] for e in batch]
    return obs, act, rew, nobs

  def add(self, obs, act, rew, nobs):
    self.memory.append((obs, act, rew, nobs))

  def size(self):
    return len(self.memory)


def get_action_ddpg(sess, network, sin, obs):
  return sess.run(network.a_out, {network.s_in: obs})


def get_action_ensemble(sess, ensemble, sin, q_res, obs):
  act_nodes = [e[0].a_out for e in ensemble]
  acts = sess.run(act_nodes, {sin: obs})

  # print(feed_dict)
  qs = []
  for i in range(num_ensemble):
    feed_dict = {sin: obs}
    v_qin = []
    for j in range(num_ensemble):
      feed_dict[ensemble[j][0].a_in] = acts[i]
    qs.append(sess.run(q_res, feed_dict))
  biggest_v = qs[0]
  biggest_i = 0
  for i in range(num_ensemble -1):
    if qs[i+1] > biggest_v:
      biggest_v = qs[i+1]
      biggest_i = i+1
  return acts[biggest_i]

register(
  id='GrlEnv-Pendulum-v0',
  entry_point='grlenv.grlenv:GrlEnv',
  kwargs={"file":"../cfg/pendulum_swingup.yaml"}
)



print("# Create Gym environment")
# Create Gym environment
env = gym.make("GrlEnv-Pendulum-v0")
print("obs--------------------______")
print(env.observation_space.shape[0])
print("# Set up Tensorflow")
# Set up Tensorflow
session = tf.Session()

print("# Create networks")
# Create networks
max_action = 3
sin = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]+1), name='s_in')
qtarget = tf.placeholder(tf.float32, shape=(None, 1), name='target')
td = tf.placeholder(tf.float32, shape=(None, num_ensemble), name='td')

ensemble = [] #DDPGNetworkEnsemble
if enable_ensemble:
  for i in range(num_ensemble):
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetworkNode(session, sin, qtarget, env.action_space.shape[0], max_action)
    target_network = DDPGNetworkNode(session, sin, qtarget, env.action_space.shape[0], max_action)
    vars = tf.trainable_variables()[prev_vars:]
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]
    ensemble.append((network, target_network, update_ops))
  #TODO refactore
  qs1 = [[ensemble[0][0].q, ensemble[1][0].q, ensemble[2][0].q]]
  qs = tf.reshape(qs1, [1,3])

  qin = tf.placeholder_with_default(tf.stop_gradient(qs), shape=(None, num_ensemble), name='qin')
  q_critic = WeightCritic(session, qin, td)
  # q_critic = average([ensemble[0][0].q, ensemble[1][0].q, ensemble[2][0].q])  # Critic merge
  # qs = [ensemble[0][0].q, ensemble[1][0].q, ensemble[2][0].q]
  # print(session.run(tf.nn.softmax(qst)))
  # sig = tf.sigmoid(qst)
  # sig_v = tf.get_variable(
  #   shape=[num_ensemble, 1], initializer=tf.truncated_normal_initializer(stddev=0.01),
  #   name="weights")
  # print(sig_v)
  # qst_l = Concatenate()(qst)
  # q_critic_d = Dense(1, activation='softmax', name='q_critic_d')(qst_l)
  # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(q_critic_d.name)[0] + '/kernel:0') + 0.0001
  # q_critic = tf.reduce_sum((qst*weights)/tf.reduce_sum(weights))
  # q_loss = tf.reduce_sum(td**2*weights/tf.reduce_sum(weights))
  # minimize_var = weights
  # r + gammaQ' - Q'
  # 64xh

  # print(q_critic_d)
  # print(weights)
  # print(q_critic)
else:
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetwork(session, env.observation_space.shape[0]+1, env.action_space.shape[0], max_action)
    target_network = DDPGNetwork(session, env.observation_space.shape[0]+1, env.action_space.shape[0], max_action)
    vars = tf.trainable_variables()[prev_vars:]
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]

# Create operations to slowly update target network
print("# Initialize weights")

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory()

print("# Run episodes")
# Run episodes
for ep in range(episodes):

  episode_reward = 0
  if (ep%10 == 0):
    test = 1
  else:
    test = 0
  observation = env.reset(test)
  observation = [math.cos(observation[0]), math.sin(observation[0]), observation[1]]

  # Loop over control steps within an episode
  noise = 0
  while True:
    # Choose action
    # action = network.get_action([observation])[0]
    if (enable_ensemble):
      # action = get_action_ensemble(session, ensemble, sin, ensemble[0][0].q, [observation])[0]
      action = get_action_ensemble(session, ensemble, sin, q_critic.q_critic, [observation])[0]
    else:
      action = get_action_ddpg(session, network, sin, [observation])[0]

    if not test:
      #action += np.random.normal(scale=[1])
      noise = 0.85 * noise + np.random.normal(scale=[1])
      action += noise

    # Take step
    prev_obs = observation
    observation, reward, done, info = env.step(action)
    observation = [math.cos(observation[0]), math.sin(observation[0]), observation[1]]

    episode_reward += reward
    reward = reward * 0.01

    # Add to replay memory
    memory.add(prev_obs, action, reward, observation)
    #print(prev_obs, ":", action, ":", reward, ":", obseration)

    if (not test):
    # Train
      if memory.size() > 1000:
        steps_in_replay = replay_steps//size_batch
        for kk in range(steps_in_replay):
          # Get minibatch from replay memory
          obs, act, rew, nobs = memory.sample_minibatch(size_batch)
          if enable_ensemble:
            ## TRAIN ACTOR CRITIC
            td_mounted = []
            qsin_mounted = []
            for ne in range(num_ensemble):
              # Calculate Q value of next state
              nextq = ensemble[ne][1].get_value(nobs)

              # Calculate target using SARSA
              target = [rew[ii] + gamma * nextq[ii] for ii in range(len(nextq))]

              # Update critic using target and actor using gradient
              ensemble[ne][0].train(obs, act, target)

              session.run(ensemble[ne][2])

              # Calculate Q value of state
              q = ensemble[ne][0].get_value(nobs)
              td_l = [target[ii] - q[ii] for ii in range(len(q))]
              if len(td_mounted) == 0:
                qsin_mounted = q
                td_mounted = td_l
              else:
                qsin_mounted = np.concatenate((qsin_mounted, q), axis=1)
                td_mounted = np.concatenate((td_mounted, td_l), axis=1)
                # qsin_mounted = np.c_[qsin_mounted, q]
                # td_mounted = np.c_[td_mounted, td_l]
            # qsin_matrix = [ qsin_mounted[ii] for ii in range(len(qsin_mounted))]
            # td_matrix = [ td_mounted[ii] for ii in range(len(td_mounted))]
            # print(qsin_matrix)
            # print(len(qsin_matrix))
            # print(td_matrix)
            q_critic.train(qsin_mounted, td_mounted)

          ## END TRAIN ACTOR CRITIC
          else:
            ## TRAIN ACTOR CRITIC
            # Calculate Q value of next state
            nextq = target_network.get_value(nobs)

            # Calculate target using SARSA
            target = [rew[ii] + gamma * nextq[ii] for ii in range(len(nextq))]

            # Update critic using target and actor using gradient
            network.train(obs, act, target)
            ## END TRAIN ACTOR CRITIC

          # Slowly update target network
          session.run(update_ops)

    if done:
      break
  if test:
    if ep > 0:
      print("          ", ep, "          ", ep*100, "          ", "{:.1f}".format(episode_reward))



