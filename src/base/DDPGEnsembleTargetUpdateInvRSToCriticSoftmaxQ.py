#FILE=base/DDPGEnsembleTargetUpdateInvRSToCriticSoftmaxQ.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE

import numpy as np

from base.Online_run import DDPGPlainEnsemble
from base.Online_run import DDPGPSoftmaxQEnsemble

# train(w/ reward_scale); train_critic(norm_td_1/rs_i) AS TargetTDError; get_action(softmax)

class DDPGEnsembleTargetUpdateInvRSToCriticSoftmaxQ(DDPGPSoftmaxQEnsemble):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleTargetUpdateInvRSToCriticSoftmaxQ")

    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        td_critic_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs, np.vstack(act))  # using minibatch action
        train_nextq_plain_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET
        acts = self.get_actions(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs) #ok
        action = self.get_action(getattr(ddpgne, "_ensemble"), getattr(ddpgne, "_sin"), nobs, q_critic.q_critic, acts, np.zeros(self._num_ensemble), q_critic.weights)
        train_nextq_critic = ddpgne.get_value(1, nobs, np.vstack(action[0]))  # get_all_actions_target_network(nobs)

        # Calculate target using SARSA
        train_target_results = []
        train_target_critic = []
        rs_critic_mounted = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_plain_results[ne][ii] for ii in range(batch_size)])
           train_target_critic.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_critic[ne][ii] for ii in range(batch_size)])
           rs_critic_mounted.append(1 / cfg_ens[ne]['reward_scale'])

        # Update critic using target and actor using gradient
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]
            td_critic_l = [train_target_critic[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                td_critic_mounted = np.abs(np.abs([rs_critic_mounted[ne] * td_critic_l[ii] for ii in range(batch_size)]))
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                td_critic_mounted = np.concatenate((td_critic_mounted, np.abs(np.abs([rs_critic_mounted[ne] * td_critic_l[ii] for ii in range(batch_size)]))), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)

        w_train = q_critic.train(td_critic_mounted, addrw_mounted, ep)

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