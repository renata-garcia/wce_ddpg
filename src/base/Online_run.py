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

    def get_actions(self, ensemble, sin, obs):
        act_nodes = [e[0].a_out for e in ensemble]
        acts =  self._session.run(act_nodes, {sin: obs})
        return acts

class DDPGSingle(OnlineRun):
    def __init__(self, sess, num_ensemble, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._print_cvs = print_cvs
        print("class DDPG_single")

    def get_action(self, network, obs): #TODO get_action equal to DDPGEnsemble
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

class DDPGPlainEnsemble(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGPlainEnsemble")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights=None): #, weights_res=None
        qss = []
        norm_q = []
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            q_crtc = self._session.run(q_res, feed_dict)
            qss.append(q_crtc)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)

        norm_q.append(qss)

        biggest_v = qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if qss[k + 1] > biggest_v:
                biggest_v = qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1

        return acts[biggest_i], [[0]], weights

class DDPGPSoftmaxQEnsemble(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGPSoftMaxQEnsemble")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights


class DDPGEnsembleRSCriticTrain(DDPGPlainEnsemble):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleRSCriticTrain")

    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        td_rs_critic_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET

        # Calculate target using SARSA
        train_target_results = []
        rs_critic_mounted = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])
           rs_critic_mounted.append(1/cfg_ens[ne]['reward_scale'])

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
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                td_rs_critic_mounted = np.abs([rs_critic_mounted[ne]*td_l[ii] for ii in range(batch_size)])
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                td_rs_critic_mounted = np.concatenate((td_rs_critic_mounted, np.abs(np.abs([rs_critic_mounted[ne]*td_l[ii] for ii in range(batch_size)]))), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)

        w_train = q_critic.train(td_rs_critic_mounted, addrw_mounted, ep)

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


class DDPGEnsembleNormV2QValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPG_ensemble")

    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []
        for iaction in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for ii in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[ii][iaction][0])
            # normalize
            norm_q_p = tmp_iaction - np.amin(tmp_iaction)
            norm_ens_qs.append(norm_q_p/(np.amax(norm_q_p) + 1e-10))
            # norm_ens_qs.append(1-np.exp(-20*norm_q_p/(np.amax(norm_q_p) + 1e-10)))

        for iq in range(self._num_ensemble):
            tmp_qs = []
            for ii in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[iq][ii])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], [[0]], weights


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
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            # normalize
            i = 0
            temp = 1
            while ((np.hstack(tmp_iaction)/temp) < -5).any() and i < 10:
                i = i + 1
                temp = temp*1e1
                if i > 10:
                    print("Online.py: tmp_iaction.any() < -5::i=10")
                    break;

            expoent = np.hstack(tmp_iaction)/temp
            norm_q.append(expoent)
            t_exp = np.exp(expoent)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)
        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights


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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxMinQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxMinQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights


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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxRSCritic(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxRSCritic")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        td_rs_critic_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET

        # Calculate target using SARSA
        train_target_results = []
        rs_critic_mounted = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])
           rs_critic_mounted.append(1 / cfg_ens[ne]['reward_scale'])

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
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                td_rs_critic_mounted = np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)])
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                td_rs_critic_mounted = np.concatenate((td_rs_critic_mounted, np.abs(np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)]))), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)

        w_train = q_critic.train(td_rs_critic_mounted, addrw_mounted, ep)

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


class DDPGEnsembleNormSoftmaxRSEtaCritic(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        self._rs_critic = []
        for ne in range(num_ensemble):
           self._rs_critic.append(1.0)
        print("class DDPGEnsembleNormSoftmaxRSEtaCritic")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            q_crtc = self._session.run(ret_dict, feed_dict)
            ens_qs.append([ self._rs_critic[ine] * q_crtc[ii] for ii in range(self._num_ensemble)])

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        td_rs_critic_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET

        # Calculate target using SARSA
        train_target_results = []
        rs_critic_mounted = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])
           rs_critic_mounted.append(1 / cfg_ens[ne]['reward_scale'])
        self._rs_critic = rs_critic_mounted

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
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                td_rs_critic_mounted = np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)])
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                td_rs_critic_mounted = np.concatenate((td_rs_critic_mounted, np.abs(np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)]))), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)

        w_train = q_critic.train(td_rs_critic_mounted, addrw_mounted, ep)

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


class DDPGEnsembleRSCriticEtaAction(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        self._rs_critic = []
        for ne in range(num_ensemble):
           self._rs_critic.append(1)
        print("class DDPGEnsembleRSCriticEtaAction")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights=None): #, weights_res=None
        qss = []
        norm_q = []
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            q_crtc = self._session.run(q_res, feed_dict)
            qss.append(self._rs_critic[ine] * q_crtc)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)

        norm_q.append(qss)

        biggest_v = qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if qss[k + 1] > biggest_v:
                biggest_v = qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1

        return acts[biggest_i], [[0]], weights


    def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ddpgne, cfg_ens, q_critic, batch_size):
        ## TRAIN ACTOR CRITIC
        td_mounted = []
        td_rs_critic_mounted = []
        q_mounted = []
        target_mounted = []

        # Calculate Q value of next state
        train_q_results = ddpgne.get_value(0, obs)
        train_nextq_results = ddpgne.get_value(1, nobs)  # TODO TD = TARGET - Q_TARGET

        # Calculate target using SARSA
        train_target_results = []
        rs_critic_mounted = []
        for ne in range(self._num_ensemble):
           train_target_results.append([rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * train_nextq_results[ne][ii] for ii in range(batch_size)])
           rs_critic_mounted.append(1 / cfg_ens[ne]['reward_scale'])
        self._rs_critic = rs_critic_mounted

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
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                td_rs_critic_mounted = np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)])
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                td_rs_critic_mounted = np.concatenate((td_rs_critic_mounted, np.abs(np.abs([rs_critic_mounted[ne] * td_l[ii] for ii in range(batch_size)]))), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)

        w_train = q_critic.train(td_rs_critic_mounted, addrw_mounted, ep)

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


class DDPGEnsembleNormSoftmaxMinMinus10TQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxMinMinus10TQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)


        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent/1e-1)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        # ok: name variable for iteration
        for i_act in range(self._num_ensemble):
            tmp_qs = []
            for iq in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[iq][i_act])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights

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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxMin10TQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxMin10TQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent/1e1)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights

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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxMin100TQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxMin100TQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent/1e2)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights

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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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


class DDPGEnsembleNormSoftmaxMin1000TQValue(OnlineRun):


    def __init__(self, sess, num_ensemble, dbg_weightstderror, print_cvs):
        self._session = sess
        self._num_ensemble = num_ensemble
        self._dbg_weightstderror = dbg_weightstderror
        self._print_cvs = print_cvs
        print("class DDPGEnsembleNormSoftmaxMin1000TQValue")


    def get_action(self, ensemble, sin, obs, q_res, acts, act_acum, weights): #, weights_res=None
        prob = []
        ens_qs = []
        ret_dict = []
        for j in range(self._num_ensemble):
            ret_dict.append(ensemble[j][0].q)

        if not (isinstance(weights, list)):
            weights = self._session.run(weights, {sin: obs})
        else:
            weights = np.hstack(weights)
        for ine in range(self._num_ensemble):
            feed_dict = {sin: obs}
            for j in range(self._num_ensemble):
                feed_dict[ensemble[j][0].a_in] = acts[ine]
            ens_qs.append(self._session.run(ret_dict, feed_dict))

        #normlize Q values
        norm_q = []
        norm_ens_qs = []
        norm_qss = []

        # q[iq,a_0] q[iq,a_1] q[iq,a_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(0)(1) q(0)(2)
        # **q(1)(0)** q(1)(1) q(1)(2)
        # **q(2)(0)** q(2)(1) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_iaction = []
            # 1 - e ^ {-10*x}
            for i_act in range(self._num_ensemble):
                tmp_iaction.append(ens_qs[i_act][iq][0])
            exponent = np.hstack(tmp_iaction) - np.max(np.hstack(tmp_iaction))
            norm_q.append(exponent)
            t_exp = np.exp(exponent/1e3)
            norm_q_p = t_exp/(np.sum(t_exp)+1e-10)
            # if np.isnan(norm_q_p[0]):
                # print("******************")

            norm_ens_qs.append(norm_q_p)

        prob.append(norm_q)

        # q[act,iq_0] q[act,iq_1] q[act,iq_2]
        # q(act)(iq) q(act)(iq) q(act)(iq)
        # **q(0)(0)** q(1)(0) q(2)(0)
        # **q(0)(1)** q(1)(1) q(2)(1)
        # **q(0)(2)** q(1)(2) q(2)(2)
        for iq in range(self._num_ensemble):
            tmp_qs = []
            for i_act in range(self._num_ensemble):
                tmp_qs.append(norm_ens_qs[i_act][iq])
            norm_qss.append(np.sum(tmp_qs*weights))

        biggest_v = norm_qss[0]
        biggest_i = 0
        for k in range(self._num_ensemble - 1):
            if norm_qss[k + 1] > biggest_v:
                biggest_v = norm_qss[k + 1]
                biggest_i = k + 1
        act_acum[biggest_i] = act_acum[biggest_i] + 1
        return acts[biggest_i], prob, weights

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
        ddpgne.train(obs, act, train_target_results)

        for ne in range(self._num_ensemble):
            # Update
            if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
                self._session.run(ddpgne._ensemble[ne][2])

            # Calculate Q value of state
            td_l = [train_target_results[ne][ii] - train_q_results[ne][ii] for ii in range(batch_size)]

            # TODO log td_l and target
            if len(td_mounted) == 0:
                q_mounted = np.abs(train_q_results)
                td_mounted = np.abs(td_l)
                target_mounted = np.abs(train_target_results)
            else:
                q_mounted = np.concatenate((q_mounted, np.abs(train_q_results)), axis=1)
                td_mounted = np.concatenate((td_mounted, np.abs(td_l)), axis=1)
                target_mounted = np.concatenate((target_mounted, np.abs(train_target_results)), axis=1)

        if self._dbg_weightstderror:
            print("FINISHED")
            print(td_mounted)
            print(target_mounted)
            print(q_mounted)
        w_train = q_critic.train(td_mounted, addrw_mounted, ep)
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
