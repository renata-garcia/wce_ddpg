#FILE=base/Online_run.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE

import itertools
import numpy as np
import tensorflow as tf


class OnlineRun:
    def __init__(self):
        self._agent = 0
        self._value_function = 0
        print("class Online_run")

    def get_policy_action(self, network, sin, obs):
        return self._session.run(network[0].a_out, {sin: obs})

class DDPGSingle(OnlineRun):
    def __init__(self, sess, num_ensemble, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._print_cvs = print_cvs
        print("class DDPG_single")

    def get_action(self, network, obs):
        return  self._session.run(network.a_out, {network.s_in: obs})

    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
            weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        # Calculate Q value of next state
        nextq = target_network.get_value(nobs)

        # Calculate target using SARSA
        target = [rew[ii] * cfg_ens[0]['reward_scale'] + cfg_ens[0]['gamma'] * nextq[ii] for ii in range(len(nextq))]

        # Update critic using target and actor using gradient
        network.train(obs, act, target)
        if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
            # Slowly update target network
            self._session.run(update_ops)

        ## END TRAIN ACTOR CRITIC

class DDPGEnsemble(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPG_ensemble")


    def get_actions(self, ensemble, sin, obs):
        act_nodes = [e[0].a_out for e in ensemble]
        acts =  self._session.run(act_nodes, {sin: obs})
        return acts


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum):
        qss = []
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
               feed_dict[ensemble[j][0].a_in] = acts[ine]
            qss.append( self._session.run(q_res, feed_dict))

        biggest_v = qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if qss[k + 1] > biggest_v:
                biggest_v = qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i]


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET
        # train_q_results = ddpgne.get_value(0, obs, act)  #using minibatch action
        # acts = self.get_actions(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs) #ok
        # action = self.get_action(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs, q_critic.q_critic, acts, np.zeros(self._num_ensemble))
        # train_nextq_results = ddpgne.get_value(1, nobs, action)  # get_all_actions_target_network(nobs)
        # # nextq = max(ensemble_q_values_target_network(nobs, actions))

        # Calculate target using SARSA
        train_target_results = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])

        # Update critic using target and actor using gradient
        # print("********************************")
        # print(obs)
        # print(act)
        # print(len(act))
        # print(train_target_results)
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = train_q_results
                td_mounted = td_l
                target_mounted = train_target_results
            else:
                q_mounted = np.concatenate((q_mounted, train_q_results), axis=1)
                td_mounted = np.concatenate((td_mounted, td_l), axis=1)
                target_mounted = np.concatenate((target_mounted, train_target_results), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted)
        weights_mounted = weights_mounted + w_train
        weights_log = np.array([w_train])
        reward_log = np.array([[reward, steps_count, ep]])
        for ne in range(batch_size - 1):
            weights_log = np.concatenate((weights_log, np.array([w_train])), axis=0)
            reward_log = np.concatenate((reward_log, np.array([[reward, steps_count, ep]])), axis=0)
        if self._print_cvs:
            data_mounted = np.concatenate((np.concatenate(
              (np.concatenate((np.concatenate((td_mounted, target_mounted), axis=1), q_mounted), axis=1), weights_log),
              axis=1),
                                           reward_log), axis=1)
            mat = np.matrix(data_mounted)
            df = pd.DataFrame(data=mat.astype(float))
            file_t = "../" + file_name + '_log.csv'
            df.to_csv(file_t, sep=' ', mode='a', header=False, float_format='%.4f', index=False)
        if self._dbg_weightstderror:
            print("axis=0")
            print(np.concatenate((np.concatenate((td_mounted, target_mounted)), q_mounted)))
            print("axis=1")
            print(data_mounted)
        return q_mounted, target_mounted, td_mounted, w_train, weights_mounted


class DDPGEnsembleTarget(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPG_ensemble")


    def get_actions(self, ensemble, sin, obs):
        act_nodes = [e[0].a_out for e in ensemble]
        acts =  self._session.run(act_nodes, {sin: obs})
        return acts


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum):
        qss = []
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            qss.append( self._session.run(q_res, feed_dict))

        biggest_v = qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if qss[k + 1] > biggest_v:
                biggest_v = qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i]


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs, np.vstack(act))  # using minibatch action
        acts = self.get_actions(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs) #ok
        action = self.get_action(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs, q_critic.q_critic, acts, np.zeros(self._num_ensemble))
        train_nextq_results = ddpgne.get_value(1, nobs, np.vstack(action))  # get_all_actions_target_network(nobs)

        # Calculate target using SARSA
        train_target_results = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])

        # Update critic using target and actor using gradient
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = train_q_results
                td_mounted = td_l
                target_mounted = train_target_results
            else:
                q_mounted = np.concatenate((q_mounted, train_q_results), axis=1)
                td_mounted = np.concatenate((td_mounted, td_l), axis=1)
                target_mounted = np.concatenate((target_mounted, train_target_results), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted)
        weights_mounted = weights_mounted + w_train
        weights_log = np.array([w_train])
        reward_log = np.array([[reward, steps_count, ep]])
        for ne in range(batch_size - 1):
            weights_log = np.concatenate((weights_log, np.array([w_train])), axis=0)
            reward_log = np.concatenate((reward_log, np.array([[reward, steps_count, ep]])), axis=0)
        if self._print_cvs:
            data_mounted = np.concatenate((np.concatenate(
              (np.concatenate((np.concatenate((td_mounted, target_mounted), axis=1), q_mounted), axis=1), weights_log),
              axis=1),
                                           reward_log), axis=1)
            mat = np.matrix(data_mounted)
            df = pd.DataFrame(data=mat.astype(float))
            file_t = "../" + file_name + '_log.csv'
            df.to_csv(file_t, sep=' ', mode='a', header=False, float_format='%.4f', index=False)
        if self._dbg_weightstderror:
            print("axis=0")
            print(np.concatenate((np.concatenate((td_mounted, target_mounted)), q_mounted)))
            print("axis=1")
            print(data_mounted)
        return q_mounted, target_mounted, td_mounted, w_train, weights_mounted


class DDPGEnsembleTDTrgt(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPG_ensemble")


    def get_actions(self, ensemble, sin, obs):
        act_nodes = [e[0].a_out for e in ensemble]
        acts =  self._session.run(act_nodes, {sin: obs})
        return acts


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum):
        qss = []
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            qss.append( self._session.run(q_res, feed_dict))

        biggest_v = qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if qss[k + 1] > biggest_v:
                biggest_v = qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i]


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state

        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET

        train_q_results_tdclc = ddpgne.get_value(0, obs, np.vstack(act))  # using minibatch action
        acts_tdclc = self.get_actions(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs) #ok
        action_tdclc = self.get_action(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs, q_critic.q_critic, acts_tdclc, np.zeros(self._num_ensemble))
        train_nextq_results_tdclc = ddpgne.get_value(1, nobs, np.vstack(action_tdclc))  # get_all_actions_target_network(nobs)

        # Calculate target using SARSA
        train_target_results = []
        train_target_results_tdclc = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])
           train_target_results_tdclc.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results_tdclc[ne][ii] for ii in range(batch_size)])

        # Update critic using target and actor using gradient
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results_tdclc[ne][ii] - train_q_results_tdclc[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = train_q_results
                td_mounted = td_l
                target_mounted = train_target_results
            else:
                q_mounted = np.concatenate((q_mounted, train_q_results), axis=1)
                td_mounted = np.concatenate((td_mounted, td_l), axis=1)
                target_mounted = np.concatenate((target_mounted, train_target_results), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted)
        weights_mounted = weights_mounted + w_train
        weights_log = np.array([w_train])
        reward_log = np.array([[reward, steps_count, ep]])
        for ne in range(batch_size - 1):
            weights_log = np.concatenate((weights_log, np.array([w_train])), axis=0)
            reward_log = np.concatenate((reward_log, np.array([[reward, steps_count, ep]])), axis=0)
        if self._print_cvs:
            data_mounted = np.concatenate((np.concatenate(
              (np.concatenate((np.concatenate((td_mounted, target_mounted), axis=1), q_mounted), axis=1), weights_log),
              axis=1),
                                           reward_log), axis=1)
            mat = np.matrix(data_mounted)
            df = pd.DataFrame(data=mat.astype(float))
            file_t = "../" + file_name + '_log.csv'
            df.to_csv(file_t, sep=' ', mode='a', header=False, float_format='%.4f', index=False)
        if self._dbg_weightstderror:
            print("axis=0")
            print(np.concatenate((np.concatenate((td_mounted, target_mounted)), q_mounted)))
            print("axis=1")
            print(data_mounted)
        return q_mounted, target_mounted, td_mounted, w_train, weights_mounted

