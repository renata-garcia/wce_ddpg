#!/usr/bin/python3
#
# OpenAI Gym example using GRL environment
# This example must be run from its own directory, with
# grl installed to a path in LD_LIBRARY_PATH and grlpy installed
# to a path in PYTHON_PATH.
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import yaml
import math, numpy as np
import base.Environments as env
import tensorflow as tf

import DDPGNetwork as dn, DDPGNetworkNode as dn, WeightCritic as wc, ReplayMemory as rm

# Register environmnent instantiation. Every configuration file
# requires a different instantiation, as Gym does not allow passing
# parameters to an environment.
# The configuration must define an "environment" tag at the root that
# specifies the environment to be used.

with open("../cfg/agent_pd_3good_j0.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

if cfg['experiment']['runs'] > 1:
  print("Experiment/runs > 1 will not save outfile file in correct manner!!!")
  exit(-1)


num_ensemble = len(cfg['experiment']['agent']['policy']['policy'])
if num_ensemble == 1:
  enable_ensemble = 1
else:
  enable_ensemble = 2
cfg_ens = []

for x in cfg['experiment']['agent']['policy']['policy']:
  str = x['representation']['file'].split()
  tmp = {}
  tmp['lr_actor'] = float(x['representation']['file'].split()[3])
  tmp['lr_critic'] = float(x['representation']['file'].split()[4])
  tmp['act1'] = x['representation']['file'].split()[5] #TODO unused
  tmp['act2'] = x['representation']['file'].split()[6] #TODO unused
  tmp['layer1'] = x['representation']['file'].split()[7] #TODO unused
  tmp['layer2'] = x['representation']['file'].split()[8] #TODO unused
  tmp['tau'] = x['representation']['tau']
  cfg_ens.append(tmp)

ii = 0
for x in cfg['experiment']['agent']['predictor']['predictor']:
  tmp = cfg_ens[ii]
  ii = ii + 1
  tmp['gamma'] = x['gamma']
  tmp['reward_scale'] = x['reward_scale'] #TODO unused

cfg_agt = {}
cfg_agt['replay_steps'] = cfg['experiment']['agent']['replay_steps']
cfg_agt['batch_size'] = cfg['experiment']['agent']['batch_size']
cfg_agt['episodes'] = cfg['experiment']['steps']  #TODO normalize


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

# register(
#   id='GrlEnv-Pendulum-v0',
#   entry_point='grlenv.grlenv:GrlEnv',
#   kwargs={"file":"../cfg/pendulum_swingup.yaml"}
# )
#
# print("# Create Gym environment")
# # Create Gym environment
# env = gym.make("GrlEnv-Pendulum-v0")
env.Environments('GrlEnv-Pendulum-v0')
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
  for ne in range(num_ensemble):
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetworkNode.DDPGNetworkNode(session, sin, qtarget, env.action_space.shape[0], max_action, cfg_ens[ne]['lr_actor'], cfg_ens[ne]['lr_critic'])
    target_network = DDPGNetworkNode.DDPGNetworkNode(session, sin, qtarget, env.action_space.shape[0], max_action, cfg_ens[ne]['lr_actor'], cfg_ens[ne]['lr_critic'])
    vars = tf.trainable_variables()[prev_vars:]
    tau = cfg_ens[ne]['tau']
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]
    ensemble.append((network, target_network, update_ops))

  qs1 = []
  for i in range(num_ensemble):
    qs1.append(ensemble[i][0].q)
  qs = tf.reshape(qs1, [1,3])

  qin = tf.placeholder_with_default(tf.stop_gradient(qs), shape=(None, num_ensemble), name='qin')
  q_critic = WeightCritic.WeightCritic(session, qin, td, 0.0001) #TODO test
else:
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetwork.DDPGNetwork(session, env.observation_space.shape[0]+1, env.action_space.shape[0], max_action, cfg_ens[0]['lr_actor'], cfg_ens[0]['lr_critic'])
    target_network = DDPGNetwork.DDPGNetwork(session, env.observation_space.shape[0]+1, env.action_space.shape[0], max_action, cfg_ens[0]['lr_actor'], cfg_ens[0]['lr_critic'])
    vars = tf.trainable_variables()[prev_vars:]
    tau = cfg_agt[0]['tau']
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]

# Create operations to slowly update target network
print("# Initialize weights")

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory.ReplayMemory()

file_name = cfg['experiment']['output']
file_output = open("../" + file_name + ".dat","w")
#TODO verificar nome do dat pros scripts

print("# Run episodes")
# Run episodes
for ep in range(cfg_agt['episodes']):

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
        steps_in_replay = cfg_agt['replay_steps']//cfg_agt['batch_size']
        for kk in range(steps_in_replay):
          # Get minibatch from replay memory
          obs, act, rew, nobs = memory.sample_minibatch(cfg_agt['batch_size'])
          if enable_ensemble:
            ## TRAIN ACTOR CRITIC
            td_mounted = []
            qsin_mounted = []
            for ne in range(num_ensemble):
              # Calculate Q value of next state
              nextq = ensemble[ne][1].get_value(nobs)

              # Calculate target using SARSA
              target = [rew[ii] + cfg_ens[ne]['gamma'] * nextq[ii] for ii in range(len(nextq))]

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
            q_critic.train(qsin_mounted, td_mounted)

          ## END TRAIN ACTOR CRITIC
          else:
            ## TRAIN ACTOR CRITIC
            # Calculate Q value of next state
            nextq = target_network.get_value(nobs)

            # Calculate target using SARSA
            target = [rew[ii] + cfg_ens[0]['gamma'] * nextq[ii] for ii in range(len(nextq))]

            # Update critic using target and actor using gradient
            network.train(obs, act, target)
            ## END TRAIN ACTOR CRITIC

          # Slowly update target network
          session.run(update_ops)

    if done:
      break
  if test:
    if ep > 0:
      log = "           %d            %d            %0.1f" % (ep, ep*100, episode_reward)
      file_output.write(log)
      print(log)
      # print("          ", ep, "          ", ep*100, "          ", "{:.1f}".format(episode_reward))


file_output.close()

