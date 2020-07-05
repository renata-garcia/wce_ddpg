#FILE=base/Online_run.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE; ./wce_ddpg.py

import numpy as np

class Online_run(): #TODO separe files and use design patterns
  def __init__(self,num_ensemble, dbg_weightstderror, print_cvs):
    self._num_ensemble = num_ensemble
    self._dbg_weightstderror = dbg_weightstderror
    self._print_cvs = print_cvs
    print("class Online_run")

    def get_action(self, sess, network, obs):
      pass

  def get_action(self):
      return self.env

class DDPG_single(Online_run):
  def __init__(self,num_ensemble, dbg_weightstderror, print_cvs):
    self._num_ensemble = num_ensemble
    self._dbg_weightstderror = dbg_weightstderror
    self._print_cvs = print_cvs
    print("class DDPG_single")

  def get_action(self, sess, network, obs):
      return sess.run(network.a_out, {network.s_in: obs})

  def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ensemble, cfg_ens, q_critic, batch_size, session):
    ## TRAIN ACTOR CRITIC
    # Calculate Q value of next state
    nextq = target_network.get_value(nobs)

    # Calculate target using SARSA
    target = [rew[ii] * cfg_ens[0]['reward_scale'] + cfg_ens[0]['gamma'] * nextq[ii] for ii in range(len(nextq))]

    # Update critic using target and actor using gradient
    network.train(obs, act, target)
    # Slowly update target network
    session.run(update_ops)
    ## END TRAIN ACTOR CRITIC

class DDPG_ensemble(Online_run):
  def __init__(self,num_ensemble, dbg_weightstderror, print_cvs):
    self._num_ensemble = num_ensemble
    self._dbg_weightstderror = dbg_weightstderror
    self._print_cvs = print_cvs
    print("class DDPG_ensemble")

  def get_action(self, sess, ensemble, sin, obs, addingreward, q_res, act_acum, addrw):
    act_nodes = [e[0].a_out for e in ensemble]
    acts = sess.run(act_nodes, {sin: obs})

    # print("addrw")
    # print(addrw)

    qss = []
    for ine in range(self._num_ensemble):
      feed_dict = {sin: obs, addingreward: addrw}
      for j in range(self._num_ensemble):
        feed_dict[ensemble[j][0].a_in] = acts[ine]
      # start_time = time.time()
      qss.append(sess.run(q_res, feed_dict))
      # print("get_action_ensemble::sess.run(q_res::--- %s seconds ---" % (time.time() - start_time))
      # print(feed_dict)
      # print("q, q_in")
      # print(sess.run(ensemble[j][0].q, feed_dict))
      # print(sess.run(q_critic._q_in, feed_dict))
    # print("add_rw_in, qin, qs")
    # print(sess.run(q_critic._adding_reward_in, feed_dict))
    # print(sess.run(q_critic.fixed, feed_dict))
    # print(qss)
    biggest_v = qss[0]
    biggest_i = 0
    for k in range(self._num_ensemble - 1):
      if qss[k + 1] > biggest_v:
        biggest_v = qss[k + 1]
        biggest_i = k + 1
    act_acum[biggest_i] = act_acum[biggest_i] + 1
    return acts[biggest_i]

  def train(self, act, addrw_mounted, ep, file_name, nobs, obs, rew, reward, steps_count,
                       weights_mounted, ensemble, cfg_ens, q_critic, batch_size, session):

      # batch_size = cfg['experiment']['agent']['batch_size']

      ## TRAIN ACTOR CRITIC
      td_mounted = []
      q_mounted = []
      target_mounted = []
      # start_time = time.time()
      for ne in range(self._num_ensemble):
        # Calculate Q value of next state
        q = ensemble[ne][0].get_value(obs)
        qtarget = ensemble[ne][1].get_value(obs)  # TODO TD = TARGET - Q_TARGET
        nextq = ensemble[ne][1].get_value(nobs)

        # Calculate target using SARSA
        target = [rew[ii] * cfg_ens[ne]['reward_scale'] + cfg_ens[ne]['gamma'] * nextq[ii] for ii in range(len(nextq))]

        # Update critic using target and actor using gradient
        ensemble[ne][0].train(obs, act, target)

        # Update
        if (steps_count % (cfg_ens[ne]['config_ddpg']._interval) == 0):
          session.run(ensemble[ne][2])
          # print(steps_count)

        # Calculate Q value of state

        td_l = [target[ii] - q[ii] for ii in range(len(q))]
        if self._dbg_weightstderror:
          print("SINGLE")
          print(td_l)
          print(target)
          print(q)

          print("BEFORE")
          print(td_mounted)
          print(target_mounted)
          print(q_mounted)

        # td_l = [(rew[ii] + cfg_ens[ne]['gamma'] * nextqactual[ii]) - q[ii] for ii in range(len(q))]
        # TODO log td_l and target
        if len(td_mounted) == 0:
          q_mounted = q
          td_mounted = td_l
          target_mounted = target
        else:
          q_mounted = np.concatenate((q_mounted, q), axis=1)
          td_mounted = np.concatenate((td_mounted, td_l), axis=1)
          target_mounted = np.concatenate((target_mounted, target), axis=1)

        if (self._dbg_weightstderror):
          print("AFTER")
          print(td_mounted)
          print(target_mounted)
          print(q_mounted)
      # end for ne in range(num_ensemble):
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


