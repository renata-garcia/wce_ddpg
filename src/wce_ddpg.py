#!/usr/bin/python3
#
# OpenAI Gym example using GRL environment
# This example must be run from its own directory, with
# grl installed to a path in LD_LIBRARY_PATH and grlpy installed
# to a path in PYTHON_PATH.

#FILE=wce_ddpg.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE; ./wce_ddpg.py

import os
import sys
import time
import random
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0"
dbg_weightstderror = 0

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# session = tf.Session(config=config)

import numpy as np
import pandas as pd
import base.Online_run as rl

from DDPGNetworkEnsemble import  DDPGNetworkEnsemble
from DDPGNetworkEnsemble import  DDPGNetworkSingle
from DDPGNetwork import  DDPGNetwork
from DDPGNetworkNode import  DDPGNetworkNode
from ReplayMemory import  ReplayMemory

from base.config import IterationMode
from base.config import ConfigYaml
from base.config import  WCE_config


def isObservationTime(memory_size):
    return memory_size < 1000


def isInitialOfEpisode(steps_count):
    return steps_count == 1


def get_action_rnd_policy(sess, network, sin, obs):
    return sess.run(network[0].a_out, {sin: obs})


def run_multi_ddpg():
    #TODO see global variable
    global ne, file_output, file_name, wce_num_ensemble
    online_iteration_mode = 0

    initial_weights = (1 / wce_num_ensemble)
    w_train = np.ones(wce_num_ensemble) * initial_weights
    weights_mounted = np.zeros((wce_num_ensemble))
    td_mounted = np.zeros((wce_num_ensemble))
    target_mounted = np.zeros((wce_num_ensemble))
    q_mounted = np.zeros((wce_num_ensemble))
    acts_diff_std = []

    steps_acum = 0
    steps_count = 1 #first init unit, reseted before use
    episode_reward = initial_weights #first init unit, reseted before use
    addrw_acum = np.ones(wce_num_ensemble, np.float32) * initial_weights
    rw_weights = np.ones(wce_num_ensemble, np.float32) * initial_weights

    for ep in range(episodes):

        act_acum = np.zeros(wce_num_ensemble)

        itmode = int(iteration_mode)
        num_rnd = random.random()
        if random.random() > 0.5:
            online_iteration_and_random_weighted_mode = 1
        else:
            online_iteration_and_random_weighted_mode = 0
        if itmode == IterationMode.alternately_persistent:
            policy_chosen = (ep % wce_num_ensemble)
        elif itmode == IterationMode.random:
            policy_chosen = int(num_rnd * wce_num_ensemble)
        elif itmode == IterationMode.random_weighted or itmode == IterationMode.policy_persistent_random_weighted: #random weighted not persistent (prob = 0.50); case not ensemble action
            chosen_arrow = num_rnd
            sum_w_pol = 0
            policy_chosen = 0
            for wt in w_train:
                sum_w_pol += wt
                if chosen_arrow > sum_w_pol:
                    policy_chosen += 1
                else:
                   break
        elif itmode == IterationMode.online:
            online_iteration_mode = 1
        elif (itmode == IterationMode.random_weighted_by_return) or (itmode == IterationMode.policy_persistent_random_weighted_by_return): #policy persistent random weighted by return
            chosen_arrow = num_rnd
            sum_w_pol = 0
            policy_chosen = 0
            for wr in rw_weights:
                sum_w_pol += wr
                if chosen_arrow > sum_w_pol:
                    policy_chosen += 1
                else:
                    break
            # print("policy_chosen: %0.1f, num_rnd: %0.2f, addrw_acum[0]: %0.2f, addrw_acum[1]: %0.2f, rw_weights[0]: %0.2f, rw_weights[1]: %0.2f" %
            #       (policy_chosen, num_rnd, addrw_acum[0], addrw_acum[1], rw_weights[0], rw_weights[1]))
        else:
            print("wce_ddpg.py::iteration_mode::", iteration_mode)
            exit(-1)

        if itmode != IterationMode.online:
            addrw_acum[policy_chosen] = addrw_acum[policy_chosen]*0.9 + 0.1*(episode_reward/steps_count)
            temperature = 1000
            if ep%wce_num_ensemble == 0 and (not isObservationTime(memory.size())):
                tmp_max = np.max(addrw_acum)
                addrw_acum_abs = addrw_acum + np.min(addrw_acum)
                e_x = np.exp(addrw_acum_abs)/temperature
                rw_weights = e_x / e_x.sum()
                # print(rw_weights)
                # tmp_min = min(addrw_acum)
                # abs_tmp_min_x_2 = abs(tmp_min*2)
                # #TODO use softmax calculating reward minus minumum
                # addrw_acum_abs = addrw_acum + abs_tmp_min_x_2
                # addrw_sum = np.sum(abs(addrw_acum_abs))
                # #TODO test increasing valur by temperature (exp/temp)
                # for ii in range(wce_num_ensemble):
                #      rw_weights[ii] = addrw_acum_abs[ii]/addrw_sum

        steps_acum = steps_acum + steps_count
        steps_count = 0
        episode_reward = 0

        test = ((ep % 10) == 0)

        observation = env.set_reset(test)
        observation = env.get_obs_trig(observation)

        # Loop over control steps within an episode
        noise = 0
        # print(test)
        while True:
            steps_count = steps_count + 1
            # Choose action

            acts = []
            tmp_ensemble = getattr(getattr(online_run, "_agent"), "_ensemble")
            if test:
                acts = online_run.get_actions(tmp_ensemble, sin, [observation])
                action = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum)[0]
            elif online_iteration_mode:
                if (random.random() < 0.05): #rnd_epsilon_action
                    action = online_run.get_policy_action(tmp_ensemble[int(random.random() * wce_num_ensemble)], sin, [observation])
                else:
                    acts = online_run.get_actions(tmp_ensemble, sin, [observation])
                    action = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum)[0]
            elif (itmode == IterationMode.policy_persistent_random_weighted) or (itmode == IterationMode.policy_persistent_random_weighted_by_return): #TODO choose by steps (not persistent)
                if (online_iteration_and_random_weighted_mode): #rnd_not_policy_persistent_random_weighted # TODO decide which policy to use for all steps, in episode
                    action = online_run.get_policy_action(tmp_ensemble[policy_chosen], sin, [observation])
                else:
                    acts = online_run.get_actions(tmp_ensemble, sin, [observation])
                    action = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum)[0]
            else:
                action = online_run.get_policy_action(tmp_ensemble[policy_chosen], sin, [observation])

            if len(acts) == 0:
                mean_acts = np.zeros(0)
            else:
                mean_acts = np.mean(acts)
            dist_acts = acts - mean_acts

            if isInitialOfEpisode(steps_count):
                dist_acts_mounted = abs(dist_acts)
                acts_diff_std = abs(np.sum(dist_acts)/wce_num_ensemble)
            else:
                dist_acts_mounted = dist_acts_mounted + abs(dist_acts)
                acts_diff_std = acts_diff_std + abs(np.sum(dist_acts)/wce_num_ensemble)

            if not test:
                #TODO read sigma for half cheetah scale=[1]
                noise = 0.85 * noise + np.random.normal(scale=[1])
                action += noise

          # Take step
            prev_obs = observation
            observation, reward, done, info = env._env.step(action)
            observation = env.get_obs_trig(observation)

            episode_reward += reward

            # Add to replay memory
            memory.add(prev_obs, action, reward, observation)

            # print("episode_reward: %d" % episode_reward)
            if (not test):
                # Train
                if not isObservationTime(memory.size()):
                    steps_in_replay = getattr(cfg_yaml, "_replay_steps") // getattr(cfg_yaml, "_batch_size")
                    for kk in range(steps_in_replay):
                        # Get minibatch from replay memory
                        obs, act, rew, nobs = memory.sample_minibatch(getattr(cfg_yaml, "_batch_size"))
                        q_mounted, target_mounted, td_mounted, w_train, weights_mounted = online_run.train(act, rw_weights, ep,
                                                                                                           file_name, nobs, obs,
                                                                                                           rew, reward,
                                                                                                           steps_count,
                                                                                                           weights_mounted,
                                                                                                           getattr(online_run, "_agent"),
                                                                                                           getattr(cfg_yaml, "_cfg_ens"),
                                                                                                           getattr(online_run, "_value_function"),
                                                                                                           getattr(cfg_yaml, "_batch_size"))
            if done:
                break
        if test:
            if ep > 1:
                log = "           %d            %d            %0.1f"\
                      % (ep, steps_acum, episode_reward)

                for ine in range(wce_num_ensemble):
                    log = log + "           %0.01f"\
                          % (weights_mounted[ine])

                if weights_mounted[0] != 0:
                    td_mounted = [sum(x) for x in zip(*td_mounted)]
                    for ine in range(wce_num_ensemble):
                         log = log + "           %0.01f"\
                               % (td_mounted[ine])

                    target_mounted_t = abs(target_mounted)
                    target_mounted_t = [sum(x) for x in zip(*target_mounted_t)]
                    for ine in range(wce_num_ensemble):
                        log = log + "           %0.01f"\
                              % (target_mounted_t[ine])

                    q_mounted_t = abs(q_mounted)
                    q_mounted_t = [sum(x) for x in zip(*q_mounted_t)]
                    for ine in range(wce_num_ensemble):
                        log = log + "           %0.01f"\
                              % (q_mounted_t[ine])
                elif ep > 1:
                    for ine in range(3*wce_num_ensemble): #3 = td, target, q
                        log = log + "           0.0"

                for iaa in range(wce_num_ensemble):
                    log = log + "           %0.01f" % (act_acum[iaa])

                for ida in range(wce_num_ensemble):
                    log = log + "           %0.01f" % (dist_acts_mounted[ida])

                log = log + "           %0.08f" % acts_diff_std

                file_output = open("../" + file_name, "a")
                file_output.write(log + "\n")
                file_output.close()

                print(log)
                weights_mounted = np.zeros((wce_num_ensemble))
                td_mounted = np.zeros((wce_num_ensemble))
                target_mounted = np.zeros((wce_num_ensemble))
                q_mounted = np.zeros((wce_num_ensemble))

          #print(observation)
          # print("          ", ep, "          ", ep*100, "          ", "{:.1f}".format(episode_reward))

# Register environmnent instantiation. Every configuration file
# requires a different instantiation, as Gym does not allow passing
# parameters to an environment.
# The configuration must define an "environment" tag at the root that
# specifies the environment to be used.

print(sys.argv)
file_yaml = sys.argv[1]
print(file_yaml)
typeCriticAggregation = sys.argv[2]
iteration_mode = sys.argv[3] #0=alternately_persistentç 1=randweightedç 2=online
run_offset = sys.argv[4]
print_cvs = int(sys.argv[5])

cfg_yaml = ConfigYaml(file_yaml)
wce_config = WCE_config()

print("# Create Gym environment")
env, steps_p_ep = wce_config.create_env(file_yaml)
max_action = env._env.action_space.high
print("min_action: ", env._env.action_space.low)
print("max_action: ", max_action)
print("obs--------------------     ", env.get_obs())

print("# Create networks")
sin = tf.placeholder(tf.float32, shape=(None, env.get_obs()), name='s_in')

print("# Set up Tensorflow")
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

wce_num_ensemble = getattr(cfg_yaml, "_num_ensemble")

print("# Set up DDPG Single or Ensemble")
if wce_num_ensemble == 1:
    online_run = rl.DDPG_single(session, wce_num_ensemble, print_cvs)
else:
    online_run = rl.DDPG_ensemble(session, wce_num_ensemble, dbg_weightstderror, print_cvs)


if getattr(cfg_yaml, "_enable_ensemble"):
    if "Target" in typeCriticAggregation:
        hasTargetActionInfo = 1
        typeCriticAggregation_ = typeCriticAggregation[6:]
    else:
        hasTargetActionInfo = 0
        typeCriticAggregation_ = typeCriticAggregation

    setattr(online_run, "_agent", DDPGNetworkEnsemble(session, sin, getattr(cfg_yaml, "_cfg_ens"),
                                                                      env._env.action_space.shape[0],
                                                                      wce_num_ensemble,
                                                                      max_action, hasTargetActionInfo))
    tmp = getattr(online_run, "_agent")
    setattr(online_run, "_value_function", tmp.get_value_function(typeCriticAggregation_))
else:
    setattr(online_run, "_agent", DDPGNetworkSingle(session, sin, getattr(cfg_yaml, "_cfg_ens"),
                                                                          env._env.action_space.shape[0],
                                                                          wce_num_ensemble,
                                                                          max_action))
    tmp = getattr(online_run, "_agent")
    setattr(online_run, "_value_function", tmp.get_value_function())


# Create operations to slowly update target network
print("# Initialize weights")

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory()

#ext =  cfg['experiment']['run_offset'] + #TODO
file_name = getattr(cfg_yaml, "_output") + typeCriticAggregation + "-" + iteration_mode + run_offset + ".txt"
file_output = open("../" + file_name, "w")
file_output.close()

if print_cvs:
    mat = np.matrix([])
    df = pd.DataFrame(data=mat.astype(float))
    file_t = "../" + file_name +'_log.csv'
    df.to_csv(file_t, sep=' ', mode='w', header=False, float_format='%.4f', index=False)

print("# Run episodes")
episodes = int(getattr(cfg_yaml, "_steps")/steps_p_ep)

random.seed(datetime.now())
# Run episodes
run_multi_ddpg()

