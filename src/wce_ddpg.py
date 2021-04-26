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

from base.DDPGEnsembleTargetSoftmaxQ import DDPGEnsembleTargetSoftmaxQ
from base.DDPGEnsembleTargetUpdateToCritic import DDPGEnsembleTargetUpdateToCritic

from DDPGNetworkEnsemble import  DDPGNetworkEnsemble
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
    global ne, file_output, file_name, wce_num_ensemble, session
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
        else:
            print("wce_ddpg.py::iteration_mode::", iteration_mode)
            exit(-1)

        if itmode != IterationMode.online:
            addrw_acum[policy_chosen] = addrw_acum[policy_chosen]*0.9 + 0.1*(episode_reward/steps_count)
            temperature = 1 #TODO 0.1 1 10
            if ep%wce_num_ensemble == 0 and (not isObservationTime(memory.size())):
                addrw_acum_abs = addrw_acum - np.min(addrw_acum)
                e_x = np.exp(addrw_acum_abs/(temperature + 1e-10))
                rw_weights = e_x / (e_x.sum() + 1e-10)

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
                action_arr, prob, test_weights = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum, getattr(online_run, "_value_function").weights) #, getattr(online_run, "_value_function").weights
                action = action_arr[0]

                mean_acts = np.mean(acts)
                dist_acts = acts - mean_acts
                dist_acts_ens = acts - action  # TODO printing

                if isInitialOfEpisode(steps_count):
                    dist_acts_mounted = abs(dist_acts)
                    dist_acts_ens_mounted = abs(dist_acts_ens)
                    acts_diff_std = abs(np.sum(dist_acts) / wce_num_ensemble)
                    acts_ens_diff_std = abs(np.sum(dist_acts_ens_mounted) / wce_num_ensemble)
                    prob_mounted = prob[0]
                else:
                    dist_acts_mounted = dist_acts_mounted + abs(dist_acts)
                    dist_acts_ens_mounted = dist_acts_ens_mounted + abs(dist_acts_ens)
                    acts_diff_std = acts_diff_std + np.sum(abs(dist_acts) / wce_num_ensemble)
                    acts_ens_diff_std = acts_ens_diff_std + np.sum(abs(dist_acts_ens_mounted) / wce_num_ensemble)
                    prob_mounted = [x + y for x, y in zip(prob_mounted, prob[0])]
            elif online_iteration_mode:
                if (random.random() < 0.05): #rnd_epsilon_action
                    action = online_run.get_policy_action(tmp_ensemble[int(random.random() * wce_num_ensemble)], sin, [observation])[0]
                else:
                    acts = online_run.get_actions(tmp_ensemble, sin, [observation])
                    action_arr, prob, test_weights = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum, getattr(online_run, "_value_function").weights)
                    action = action_arr[0]
            elif (itmode == IterationMode.policy_persistent_random_weighted) or (itmode == IterationMode.policy_persistent_random_weighted_by_return): #TODO choose by steps (not persistent)
                if (online_iteration_and_random_weighted_mode): #rnd_not_policy_persistent_random_weighted # TODO decide which policy to use for all steps, in episode
                    action = online_run.get_policy_action(tmp_ensemble[policy_chosen], sin, [observation])[0]
                else:
                    acts = online_run.get_actions(tmp_ensemble, sin, [observation])
                    action_arr, prob, test_weights = online_run.get_action(tmp_ensemble, sin, [observation], getattr(online_run, "_value_function").q_critic, acts, act_acum, getattr(online_run, "_value_function").weights)
                    action = action_arr[0]
            else:
                action = online_run.get_policy_action(tmp_ensemble[policy_chosen], sin, [observation])[0]


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
            # ep, steps_acum, episode_reward, #weights_mounted #td_mounted_t #target_mounted_t #q_mounted_t
            if ep > 1:
                log = "%d\t%d\t%0.1f"\
                      % (ep, steps_acum, episode_reward)

                for ine in range(wce_num_ensemble):
                    log = log + "\t%0.01f"\
                          % (weights_mounted[ine])

                if weights_mounted[0] != 0:
                    td_mounted_t = [sum(x) for x in zip(*td_mounted)]
                    for ine in range(wce_num_ensemble):
                         log = log + "\t%0.01f"\
                               % (td_mounted_t[ine])

                    target_mounted_t = [sum(x) for x in zip(*target_mounted)]
                    for ine in range(wce_num_ensemble):
                        log = log + "\t%0.01f"\
                              % (target_mounted_t[ine])

                    q_mounted_t = [sum(x) for x in zip(*q_mounted)]
                    for ine in range(wce_num_ensemble):
                        log = log + "\t%0.01f"\
                              % (q_mounted_t[ine])
                elif ep > 1:
                    for ine in range(3*wce_num_ensemble): #3 = td, target, q
                        log = log + "\t0.0"

                for iaa in range(wce_num_ensemble):
                    log = log + "\t%0.01f" % (act_acum[iaa])

                for ida in range(wce_num_ensemble):
                    if len(dist_acts_mounted[ida][0]) == 1:
                        log = log + "\t%0.01f" % (dist_acts_mounted[ida])
                    else:
                        for jda in range(len(dist_acts_mounted[ida][0])):
                            if len(dist_acts_mounted[ida][0]) > 1:
                                log = log + "\t%0.01f" % (dist_acts_mounted[ida][0][jda])
                            else:
                                log = log + "\t%0.01f" % (dist_acts_mounted[ida][jda])

                log = log + "\t%0.01f" % acts_diff_std

                for ida in range(wce_num_ensemble):
                    if len(dist_acts_mounted[ida][0]) == 1:
                        log = log + "\t%0.01f" % (dist_acts_ens_mounted[ida])
                    else:
                        for jda in range(len(dist_acts_ens_mounted[ida][0])):
                            if len(dist_acts_ens_mounted[ida][0]) > 1:
                                log = log + "\t%0.01f" % (dist_acts_ens_mounted[ida][0][jda])
                            else:
                                log = log + "\t%0.01f" % (dist_acts_ens_mounted[ida][jda])

                log = log + "\t%0.01f" % acts_ens_diff_std

                if len(np.hstack(prob_mounted)) > 1:
                    for ipb in range(len(np.hstack(prob_mounted))):
                        log = log + "\t%0.01f" % (np.hstack(prob_mounted)[ipb])
                else:
                    for ipb in range(wce_num_ensemble*wce_num_ensemble):
                        log = log + "\t0.0"

                for itw in range(wce_num_ensemble):
                    log = log + "\t%0.07f" % test_weights[itw]

                file_output = open("../" + file_name, "a")
                file_output.write(log + "\n")
                file_output.close()

                print(log)
                weights_mounted = np.zeros((wce_num_ensemble))
                td_mounted = np.zeros((wce_num_ensemble))
                target_mounted = np.zeros((wce_num_ensemble))
                q_mounted = np.zeros((wce_num_ensemble))
                prob_mounted = np.zeros((wce_num_ensemble))
                dist_acts_mounted = np.zeros((wce_num_ensemble))
                acts_diff_std = np.zeros((wce_num_ensemble))
                dist_acts_ens_mounted = np.zeros((wce_num_ensemble))
                acts_ens_diff_std = np.zeros((wce_num_ensemble))
                prob_mounted = []

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
adding_name = sys.argv[6]
fool_ps_name = sys.argv[7]

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

if "Target" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[6:]
    online_run = rl.DDPGEnsembleTarget(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "TrgtSoftmaxQ" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[12:]
    online_run = DDPGEnsembleTargetSoftmaxQ(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "TrgtUpdateToCritic" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[18:]
    online_run = DDPGEnsembleTargetUpdateToCritic(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "TDTrgt" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[6:]
    online_run = rl.DDPGEnsembleTDTrgt(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[17:]
    online_run = rl.DDPGEnsembleNormSoftmaxQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxMinQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[20:]
    online_run = rl.DDPGEnsembleNormSoftmaxMinQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxMinMinus10TQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[28:]
    online_run = rl.DDPGEnsembleNormSoftmaxMinMinus10TQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxMin10TQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[23:]
    online_run = rl.DDPGEnsembleNormSoftmaxMin10TQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxMin100TQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[24:]
    online_run = rl.DDPGEnsembleNormSoftmaxMin100TQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxMin1000TQValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[25:]
    online_run = rl.DDPGEnsembleNormSoftmaxMin1000TQValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormV2QValue" in typeCriticAggregation:
    typeCriticAggregation_ = typeCriticAggregation[12:]
    online_run = rl.DDPGEnsembleNormV2QValue(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "RSCriticTrain" in typeCriticAggregation: #train(w/ reward_scale); train_critic(norm_1/rs_i); get_action(plain)
    typeCriticAggregation_ = typeCriticAggregation[13:]
    online_run = rl.DDPGEnsembleRSCriticTrain(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxRSCritic" in typeCriticAggregation: #train(w/ reward_scale); train_critic(norm_td_1/rs_i); get_action(softmax)
    typeCriticAggregation_ = typeCriticAggregation[19:]
    online_run = rl.DDPGEnsembleNormSoftmaxRSCritic(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "RSCriticEtaAction" in typeCriticAggregation: #train(w/ reward_scale); train_critic(norm_1/rs_i); get_action(norm_q_1/rs_i)
    typeCriticAggregation_ = typeCriticAggregation[17:]
    online_run = rl.DDPGEnsembleRSCriticEtaAction(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
elif "NormSoftmaxRSEtaCritic" in typeCriticAggregation: #train(w/ reward_scale); train_critic(norm_1/rs_i); get_action(softmax_of_norm_q_1/rs_i)
    typeCriticAggregation_ = typeCriticAggregation[22:]
    online_run = rl.DDPGEnsembleNormSoftmaxRSEtaCritic(session, wce_num_ensemble, dbg_weightstderror, print_cvs)
else:
    typeCriticAggregation_ = typeCriticAggregation
    online_run = rl.DDPGEnsemble(session, wce_num_ensemble, dbg_weightstderror, print_cvs)

setattr(online_run, "_agent", DDPGNetworkEnsemble(session, sin, getattr(cfg_yaml, "_cfg_ens"),
                                                                  env._env.action_space.shape[0],
                                                                  wce_num_ensemble,
                                                                  max_action))
tmp = getattr(online_run, "_agent")
setattr(online_run, "_value_function", tmp.build_value_function(typeCriticAggregation_))

# Create operations to slowly update target network
print("# Initialize weights")

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory()

#ext =  cfg['experiment']['run_offset'] + #TODO
file_name = getattr(cfg_yaml, "_output") + typeCriticAggregation + "-" + iteration_mode + run_offset + adding_name[1:] + ".txt"
print(file_name)
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

