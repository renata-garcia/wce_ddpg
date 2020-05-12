#!/usr/bin/python3
#
# OpenAI Gym example using GRL environment
# This example must be run from its own directory, with
# grl installed to a path in LD_LIBRARY_PATH and grlpy installed
# to a path in PYTHON_PATH.

#FILE=wce_ddpg.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE; ./wce_ddpg.py

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import yaml
import numpy as np
import base.Environments as be
import base.DDPGNetworkConfig as ddpg_cfg
import tensorflow as tf
from random import random
from random import seed

import DDPGNetwork, DDPGNetworkNode, ReplayMemory
from CriticAggregation import WeightedByTDError
from CriticAggregation import WeightedByAverage

def get_action_ddpg(sess, network, obs):
  rnd_policy = random()
  if (rnd_policy < 0.05):
      return (2*rnd_policy - 1 ) * max_action
  else:
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


def get_action_rnd_policy(sess, network, sin, obs):
  return sess.run(network[0].a_out, {sin: obs})


def run_multi_ddpg():
  global ne, file_output
  w_train = 0
  weights_mounted = np.zeros((num_ensemble))
  steps_acum = 0
  steps_count = 0
  for ep in range(episodes):

    steps_acum = steps_acum + steps_count
    steps_count = 0
    episode_reward = 0

    if (ep % 10 == 0):
      test = 1
      policy_rnd = int(random() * num_ensemble)
      # print(policy_rnd)
      # print(ep)
    else:
      test = 0
    #observation = env._env.reset(test) TODO refactore
    observation = env._env.reset()
    observation = env.get_obs_trig(observation)

    # Loop over control steps within an episode
    noise = 0
    while True:
      steps_count = steps_count + 1
      # Choose action
      if enable_ensemble:
        if test:
          action = get_action_ensemble(session, ensemble, sin, q_critic.q_critic, [observation])[0]
        else:
          action = get_action_rnd_policy(session, ensemble[policy_rnd], sin, [observation])[0]
      else:
        action = get_action_ddpg(session, network, sin, [observation])[0]

      if not test:
        noise = 0.85 * noise + np.random.normal(scale=[1])
        action += noise

      # Take step
      prev_obs = observation
      observation, reward, done, info = env._env.step(action)
      observation = env.get_obs_trig(observation)

      episode_reward += reward
      reward = reward * 0.01 #TODO reward_scale

      # Add to replay memory
      memory.add(prev_obs, action, reward, observation)
      #print(observation)

      if (not test):
        # Train
        if memory.size() > 1000:

          # print("weights_mounted")
          # print(weights_mounted)

          steps_in_replay = replay_steps // batch_size
          for kk in range(steps_in_replay):
            # Get minibatch from replay memory
            obs, act, rew, nobs = memory.sample_minibatch(batch_size)
            if enable_ensemble:
              ## TRAIN ACTOR CRITIC
              td_mounted = []
              qsin_mounted = []
              for ne in range(num_ensemble):
                # Calculate Q value of next state
                q = ensemble[ne][0].get_value(nobs)
                nextq = ensemble[ne][1].get_value(nobs)

                # Calculate target using SARSA
                target = [rew[ii] + cfg_ens[ne]['gamma'] * nextq[ii] for ii in range(len(nextq))]

                # Update critic using target and actor using gradient
                ensemble[ne][0].train(obs, act, target)

                # Update
                session.run(ensemble[ne][2])

                # Calculate Q value of state
                td_l = [target[ii] - q[ii] for ii in range(len(q))]
                if len(td_mounted) == 0:
                  qsin_mounted = q
                  td_mounted = td_l
                else:
                  qsin_mounted = np.concatenate((qsin_mounted, q), axis=1)
                  td_mounted = np.concatenate((td_mounted, td_l), axis=1)
              w_train = q_critic.train(td_mounted) #qsin_mounted, td_mounted) TODO refactore
              weights_mounted = weights_mounted + w_train

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
        # log = "           %d            %d            %0.1f" % (ep, steps_acum, episode_reward)

        if (num_ensemble == 2):
          log = "           %d            %d            %0.1f           %0.01f            %0.01f" % (ep, steps_acum, episode_reward, weights_mounted[0], weights_mounted[1])
        elif (num_ensemble == 3):
          log = "           %d            %d            %0.1f           %0.01f            %0.01f" % (ep, steps_acum, episode_reward, weights_mounted[0], weights_mounted[1], weights_mounted[2])

        file_output = open("../" + file_name, "a")
        file_output.write(log + "\n")
        file_output.close()

        print(log)

        weights_mounted = np.zeros((num_ensemble))

        #print(observation)
        # print("          ", ep, "          ", ep*100, "          ", "{:.1f}".format(episode_reward))

def create_env():
  steps_p_ep = 0
  name_print = "wce_ddpg.py::create_env()::"
  if "pd" in file_yaml:
    env = be.Environments('GrlEnv-Pendulum-v0')
    steps_p_ep = 100
    print(name_print, "GrlEnv-Pendulum-v0")
  elif "cp" in file_yaml:
    env = be.Environments('GrlEnv-CartPole-v0')
    steps_p_ep = 200
    print(name_print, "GrlEnv-CartPole-v0")
  elif "cdp" in file_yaml:
    env = be.Environments('GrlEnv-CartDoublePole-v0')
    steps_p_ep = 200
    print(name_print, "GrlEnv-CartDoublePole-v0")
  elif "_hc_" in file_yaml:
    env = be.Environments('GrlEnv-HalfCheetah-v2')
    steps_p_ep = 1000
    print(name_print, "GrlEnv-HalfCheetah-v2")
  elif "_r_" in file_yaml:
    env = be.Environments('Gym-Reacher-v2')
    steps_p_ep = 500
    print(name_print, "Gym-Reacher-v2")
  elif "_hs_" in file_yaml:
    env = be.Environments('Gym-HumanoidStandup-v2')
    steps_p_ep = 1000
    print(name_print, "Gym-HumanoidStandup-v2")
  elif "_cr_" in file_yaml:
    env = be.Environments('Gym-CarRacing-v0')
    steps_p_ep = 1000
    print(name_print, "Gym-CarRacing-v0")
  elif "_humanoid_" in file_yaml:
    env = be.Environments('Gym-Humanoid-v2')
    steps_p_ep = 1000
    print(name_print, "Gym-Humanoid-v2")
  elif "_ant_" in file_yaml:
    env = be.Environments('Gym-Ant-v2')
    steps_p_ep = 1000
    print(name_print, "Gym-Ant-v2")
  else:
    print(name_print, file_yaml)
    exit(-1)
  return env, steps_p_ep


# Register environmnent instantiation. Every configuration file
# requires a different instantiation, as Gym does not allow passing
# parameters to an environment.
# The configuration must define an "environment" tag at the root that
# specifies the environment to be used.

print(sys.argv)
file_yaml = sys.argv[1]
print(file_yaml)
typeCriticAgregattion = sys.argv[2] #TODO make it robust "Average"
with open(file_yaml, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

if cfg['experiment']['runs'] > 1:
  print("Experiment/runs > 1 will not save outfile file in correct manner!!!")
  exit(-1)


num_ensemble = len(cfg['experiment']['agent']['policy']['policy'])
if num_ensemble == 1:
  enable_ensemble = 0
else:
  enable_ensemble = 1
print("num_ensemble:")
print(num_ensemble)
cfg_ens = []


for x in cfg['experiment']['agent']['policy']['policy']:
  str = x['representation']['file'].split()
  tmp = {}
  tmp['lr_actor'] = float(x['representation']['file'].split()[3])
  tmp['lr_critic'] = float(x['representation']['file'].split()[4])
  tmp['act1'] = x['representation']['file'].split()[5]
  tmp['act2'] = x['representation']['file'].split()[6]
  tmp['layer1'] = int(x['representation']['file'].split()[7])
  tmp['layer2'] = int(x['representation']['file'].split()[8])
  tmp['tau'] =  float(x['representation']['tau'])
  tmp['config_ddpg'] = ddpg_cfg.DDPGNetworkConfig(tmp['lr_actor'], tmp['lr_critic'], tmp['act1'], tmp['act2'], tmp['layer1'], tmp['layer2'], tmp['tau'])
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
cfg_agt['steps'] = cfg['experiment']['steps']  #TODO normalize
cfg_agt['run_offset'] = cfg['experiment']['run_offset']


print("# Create Gym environment")
# # Create Gym environment
#env._env.step(action)
env, steps_p_ep = create_env()

max_action = env._env.action_space.high
print("min_action: ", env._env.action_space.low)
print("max_action: ", max_action)
print("obs--------------------     ", env.get_obs())

print("# Set up Tensorflow")
# Set up Tensorflow
session = tf.Session()

print("# Create networks")
# Create networks
sin = tf.placeholder(tf.float32, shape=(None, env.get_obs()), name='s_in')
qtarget = tf.placeholder(tf.float32, shape=(None, 1), name='target')
td = tf.placeholder(tf.float32, shape=(None, num_ensemble), name='td')


ensemble = [] #DDPGNetworkEnsemble
if enable_ensemble:
  for ne in range(num_ensemble):
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetworkNode.DDPGNetworkNode(session, sin, qtarget, env._env.action_space.shape[0], max_action, cfg_ens[ne]['config_ddpg']) #cfg_ens[ne]['lr_actor'], cfg_ens[ne]['lr_critic'])
    target_network = DDPGNetworkNode.DDPGNetworkNode(session, sin, qtarget, env._env.action_space.shape[0], max_action, cfg_ens[ne]['config_ddpg']) #, cfg_ens[ne]['lr_actor'], cfg_ens[ne]['lr_critic'])
    vars = tf.trainable_variables()[prev_vars:]
    tau = cfg_ens[ne]['tau']
    #TODO dividir o tau pelo interval....
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]
    ensemble.append((network, target_network, update_ops))
    #print("Create network ne:", ne, ", lr_actor: ", cfg_ens[ne]['lr_actor'],  ", lr_critic: ", cfg_ens[ne]['lr_critic'])

  qs1 = []
  for i in range(num_ensemble):
    qs1.append(ensemble[i][0].q)
  qs = tf.reshape(qs1, [1,num_ensemble])

  qin = tf.placeholder_with_default(tf.stop_gradient(qs), shape=(None, num_ensemble), name='qin')
  #q_critic = WeightCritic.WeightCritic(session, qin, td, num_ensemble)
  if typeCriticAgregattion == "Average":
    q_critic = WeightedByAverage(session, qs1, td, num_ensemble)
  else:
    q_critic = WeightedByTDError(session, qin, td, num_ensemble)
  q_critic.buildLayer()

else:
    prev_vars = len(tf.trainable_variables())
    network = DDPGNetwork.DDPGNetwork(session, env._env.observation_space.shape[0]+1, env._env.action_space.shape[0], max_action, cfg_ens[0]['config_ddpg']) #, cfg_ens[0]['lr_actor'], cfg_ens[0]['lr_critic'])
    target_network = DDPGNetwork.DDPGNetwork(session, env._env.observation_space.shape[0]+1, env._env.action_space.shape[0], max_action, cfg_ens[0]['config_ddpg']) #, cfg_ens[0]['lr_actor'], cfg_ens[0]['lr_critic'])
    vars = tf.trainable_variables()[prev_vars:]
    tau = cfg_ens[0]['tau']
    update_ops = [vars[ix + len(vars) // 2].assign_add(tau * (var.value() - vars[ix + len(vars) // 2].value())) for
                  ix, var in enumerate(vars[0:len(vars) // 2])]


# Create operations to slowly update target network
print("# Initialize weights")

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory.ReplayMemory()


#ext =  cfg['experiment']['run_offset'] + #TODO
file_name = cfg['experiment']['output'] + typeCriticAgregattion + "-" +  ".txt"
file_output = open("../" + file_name, "w")
file_output.close()
#TODO verificar nome do dat pros scripts

print("# Run episodes")
episodes = int(cfg_agt['steps']/steps_p_ep)
replay_steps = cfg_agt['replay_steps']
batch_size = cfg_agt['batch_size']
reward_scale = cfg_ens[0]['reward_scale']

seed(1234)

# Run episodes

run_multi_ddpg()
