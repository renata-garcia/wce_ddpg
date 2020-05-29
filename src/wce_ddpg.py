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
dbg_weightstderror = 0

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

import yaml
import numpy as np
import pandas as pd
import base.Environments as be
import base.DDPGNetworkConfig as ddpg_cfg
from random import random
from random import seed

import DDPGNetwork, DDPGNetworkNode, ReplayMemory
from CriticAggregation import WeightedByTDError
from CriticAggregation import WeightedByAverage
from CriticAggregation import WeightedByTDErrorInvW
from CriticAggregation import WeightedByTDErrorAddingReward
from CriticAggregation import WeightedByAddingReward

def get_action_ddpg(sess, network, obs):
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
  global ne, file_output, file_name

  w_train = 0
  weights_mounted = np.zeros((num_ensemble))
  td_mounted = np.zeros((num_ensemble))
  target_mounted = np.zeros((num_ensemble))
  q_mounted = np.zeros((num_ensemble))

  steps_acum = 0
  steps_count = 1 #first init unit, reseted before use
  episode_reward = 1 #first init unit, reseted before use
  addrw_acum = np.zeros(num_ensemble, np.float32)
  for ep in range(episodes):

    #TODO selecionar a politica alternadamente
    #policy_rnd = int(random() * num_ensemble)
    policy_rnd = (ep%num_ensemble)
    # print("ep %d, policy_rnd %d, num_ensemble %d" % (ep, policy_rnd, num_ensemble))

    addrw_mounted = np.zeros(num_ensemble, np.float32)
    addrw_acum[policy_rnd] += episode_reward/steps_count
    addrw_sum = np.sum(addrw_acum)
    for ii in range(num_ensemble):
      addrw_mounted[ii] = addrw_acum[ii]/addrw_sum
    # print("addrw_mounted")
    # print(addrw_mounted)

    steps_acum = steps_acum + steps_count
    steps_count = 0
    episode_reward = 0

    if (ep % 10 == 0):
      test = 1
    else:
      test = 0

    observation = env.set_reset(test)
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

          steps_in_replay = replay_steps // batch_size
          for kk in range(steps_in_replay):
            # Get minibatch from replay memory
            obs, act, rew, nobs = memory.sample_minibatch(batch_size)
            if enable_ensemble:
              ## TRAIN ACTOR CRITIC
              td_mounted = []
              q_mounted = []
              target_mounted = []
              for ne in range(num_ensemble):
                # Calculate Q value of next state
                q = ensemble[ne][0].get_value(obs)
                qtarget = ensemble[ne][1].get_value(obs) #TODO TD = TARGET - Q_TARGET
                nextq = ensemble[ne][1].get_value(nobs)

                # Calculate target using SARSA
                target = [rew[ii] + cfg_ens[ne]['gamma'] * nextq[ii] for ii in range(len(nextq))]

                # Update critic using target and actor using gradient
                ensemble[ne][0].train(obs, act, target)

                # Update
                session.run(ensemble[ne][2])

                # Calculate Q value of state
                td_l = [ target[ii] - q[ii] for ii in range(len(q)) ]
                if dbg_weightstderror:
                  print("SINGLE")
                  print(td_l)
                  print(target)
                  print(q)

                  print("BEFORE")
                  print(td_mounted)
                  print(target_mounted)
                  print(q_mounted)

                # td_l = [(rew[ii] + cfg_ens[ne]['gamma'] * nextqactual[ii]) - q[ii] for ii in range(len(q))]
                #TODO log td_l and target
                if len(td_mounted) == 0:
                  q_mounted = q
                  td_mounted = td_l
                  target_mounted = target
                else:
                  q_mounted = np.concatenate((q_mounted, q), axis=1)
                  td_mounted = np.concatenate((td_mounted, td_l), axis=1)
                  target_mounted = np.concatenate((target_mounted, target), axis=1)

                if(dbg_weightstderror):
                  print("AFTER")
                  print(td_mounted)
                  print(target_mounted)
                  print(q_mounted)

              if dbg_weightstderror:
                print("FINISHED")
                print(td_mounted)
                print(target_mounted)
                print(q_mounted)

              w_train = q_critic.train(td_mounted, addrw_mounted)

              weights_mounted = weights_mounted + w_train

              weights_log = np.array([w_train])
              reward_log = np.array([[reward, steps_count, ep]])
              for ne in range(batch_size-1):
                weights_log = np.concatenate((weights_log, np.array([w_train])), axis=0)
                reward_log = np.concatenate((reward_log,  np.array([[reward, steps_count, ep]])), axis=0)

              if print_cvs:
                data_mounted = np.concatenate((np.concatenate((np.concatenate((np.concatenate((td_mounted, target_mounted), axis=1), q_mounted), axis=1), weights_log), axis=1), reward_log), axis=1)
                mat = np.matrix(data_mounted)
                df = pd.DataFrame(data=mat.astype(float))
                file_t = "../" + file_name + '_log.csv'
                df.to_csv(file_t, sep=' ', mode='a', header=False, float_format='%.4f', index=False)

              if dbg_weightstderror:
                print("axis=0")
                print(np.concatenate((np.concatenate((td_mounted, target_mounted)), q_mounted)))
                print("axis=1")
                print(data_mounted)

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
            # act = session.run(ensemble[0][0].a_out, {sin: [observation]})
            # session.run(q_critic.q_critic, {ensemble[0][0].a_in: act})
            # print("q_critic.q_critic")
            # print(q_critic.q_critic)

      if done:
        break
    if test:
      if ep > 1:
        log = "           %d            %d            %0.1f" \
                % (ep, steps_acum, episode_reward)

        for ine in range(num_ensemble):
          log = log + "           %0.01f" \
                % (weights_mounted[ine])

        if weights_mounted[0] != 0:
          td_mounted = [sum(x) for x in zip(*td_mounted)]
          for ine in range(num_ensemble):
            log = log + "           %0.01f" \
                  % (td_mounted[ine])

          target_mounted_t = abs(target_mounted)
          target_mounted_t = [sum(x) for x in zip(*target_mounted_t)]
          for ine in range(num_ensemble):
            log = log + "           %0.01f" \
                  % (target_mounted_t[ine])

          q_mounted_t = abs(q_mounted)
          q_mounted_t = [sum(x) for x in zip(*q_mounted_t)]
          for ine in range(num_ensemble):
            log = log + "           %0.01f" \
                  % (q_mounted_t[ine])
        elif ep > 1:
          for ine in range(3*num_ensemble): #3 = td, target, q
            log = log + "           0.0"

        file_output = open("../" + file_name, "a")
        file_output.write(log + "\n")
        file_output.close()

        print(log)
        weights_mounted = np.zeros((num_ensemble))
        td_mounted = np.zeros((num_ensemble))
        target_mounted = np.zeros((num_ensemble))
        q_mounted = np.zeros((num_ensemble))

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
typeCriticAggregation = sys.argv[2]
run_offset = sys.argv[3]
print_cvs = int(sys.argv[4])
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
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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

  addrw = np.zeros(num_ensemble, dtype=np.float32)
  addingreward = tf.placeholder_with_default(tf.stop_gradient(addrw), shape=(num_ensemble), name='addingreward')
  # addingreward = tf.placeholder(tf.float32, shape=(num_ensemble), name='addingreward')
  print("addingreward")
  print(addingreward)

  if typeCriticAggregation == "Average":
    q_critic = WeightedByAverage(session, qs1, td, num_ensemble)
  elif typeCriticAggregation == "TDErrorInvW":
    q_critic = WeightedByTDErrorInvW(session, qin, td, num_ensemble)
  elif typeCriticAggregation == "TDErrorAddRw":
    q_critic = WeightedByTDErrorAddingReward(session, qin, td, num_ensemble, addingreward)
  elif typeCriticAggregation == "AddRw":
    q_critic = WeightedByAddingReward(session, qin, td, num_ensemble, addingreward)
  elif typeCriticAggregation == "TDError":
    q_critic = WeightedByTDError(session, qin, td, num_ensemble)
  else:
    print("typeCriticAggregation")
    print(typeCriticAggregation)
    exit(-1)

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
file_name = cfg['experiment']['output'] + typeCriticAggregation + "-" + run_offset + ".txt"
file_output = open("../" + file_name, "w")
file_output.close()

if print_cvs:
  mat = np.matrix([])
  df = pd.DataFrame(data=mat.astype(float))
  file_t = "../" + file_name +'_log.csv'
  df.to_csv(file_t, sep=' ', mode='w', header=False, float_format='%.4f', index=False)

print("# Run episodes")
episodes = int(cfg_agt['steps']/steps_p_ep)
replay_steps = cfg_agt['replay_steps']
batch_size = cfg_agt['batch_size']
reward_scale = cfg_ens[0]['reward_scale']

seed(1234)

# Run episodes

run_multi_ddpg()

