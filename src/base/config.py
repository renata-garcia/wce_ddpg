import yaml

from base.Environments import Environments

class ConfigYaml:

    def __init__(self, file_yaml):
        self._cfg_ens = []
        self._cfg_agt = {}
        self._file_yaml = file_yaml
        with open(self._file_yaml, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile)

        if self._cfg['experiment']['runs'] > 1:
            print("Experiment/runs > 1 will not save outfile file in correct manner!!!")
            exit(-1)
        self._enable_ensemble = (len(self._cfg['experiment']['agent']['policy']) == 11)
        if self._enable_ensemble:
            self._num_ensemble = 1
        else:
            self._num_ensemble = len(self._cfg['experiment']['agent']['policy']['policy'])
        self.read_file()
        self._replay_steps = self._cfg['experiment']['agent']['replay_steps']
        self._batch_size = self._cfg['experiment']['agent']['batch_size']
        self._steps = self._cfg['experiment']['steps']
        self._run_offset = self._cfg['experiment']['run_offset']
        self._output = self._cfg['experiment']['output']


    def read_file(self):
        if self._enable_ensemble:
            x = self._cfg['experiment']['agent']['policy']
            str = x['representation']['file'].split()
            tmp = {'lr_actor': float(x['representation']['file'].split()[3]),
                   'lr_critic': float(x['representation']['file'].split()[4]),
                   'act1': x['representation']['file'].split()[5],
                   'act2': x['representation']['file'].split()[6],
                   'layer1': int(x['representation']['file'].split()[7]),
                   'layer2': int(x['representation']['file'].split()[8]),
                   'tau': float(x['representation']['tau']),
                   'interval': float(x['representation']['interval']),
                   }
            tmp['config_ddpg'] = DDPGNetworkConfig(tmp['lr_actor'], tmp['lr_critic'], tmp['act1'], tmp['act2'],
                                                   tmp['layer1'], tmp['layer2'], tmp['tau'], tmp['interval'])
            self._cfg_ens.append(tmp)
            x = self._cfg['experiment']['agent']['predictor']
            tmp = self._cfg_ens[0]
            tmp['gamma'] = x['gamma']
            tmp['reward_scale'] = x['reward_scale']
            tmp['config_ddpg'].setGamma(tmp['gamma'])
            tmp['config_ddpg'].setRWScale(tmp['reward_scale'])
        else:
            policy = self._cfg['experiment']['agent']['policy']['policy']
            for x in policy:
                #TODO
                str = x['representation']['file'].split()
                tmp = {'lr_actor': float(x['representation']['file'].split()[3]),
                       'lr_critic': float(x['representation']['file'].split()[4]),
                       'act1': x['representation']['file'].split()[5],
                       'act2': x['representation']['file'].split()[6],
                       'layer1': int(x['representation']['file'].split()[7]),
                       'layer2': int(x['representation']['file'].split()[8]),
                       'tau': float(x['representation']['tau']),
                       'interval': float(x['representation']['interval']),
                       }
                tmp['config_ddpg'] = DDPGNetworkConfig(tmp['lr_actor'], tmp['lr_critic'], tmp['act1'], tmp['act2'],
                                                                tmp['layer1'], tmp['layer2'], tmp['tau'], tmp['interval'])
                self._cfg_ens.append(tmp)

            ii = 0
            for x in self._cfg['experiment']['agent']['predictor']['predictor']:
                tmp = self._cfg_ens[ii]
                ii = ii + 1
                tmp['gamma'] = x['gamma']
                tmp['reward_scale'] = x['reward_scale']
                tmp['config_ddpg'].setGamma(tmp['gamma'])
                tmp['config_ddpg'].setRWScale(tmp['reward_scale'])


class  WCE_config:

    def __init__(self):
        print("WCE_config")


    def create_env(self, file_yaml):
        steps_p_ep = 0
        name_print = "wce_ddpg.py::create_env()::"
        if "pd" in file_yaml:
            env = Environments('GrlEnv-Pendulum-v0')
            steps_p_ep = 100
            print(name_print, "GrlEnv-Pendulum-v0")
        elif "cp" in file_yaml:
            env = Environments('GrlEnv-CartPole-v0')
            steps_p_ep = 200
            print(name_print, "GrlEnv-CartPole-v0")
        elif "cdp" in file_yaml:
            env = Environments('GrlEnv-CartDoublePole-v0')
            steps_p_ep = 200
            print(name_print, "GrlEnv-CartDoublePole-v0")
        elif "_hc_" in file_yaml:
            env = Environments('GrlEnv-HalfCheetah-v2')
            steps_p_ep = 1000
            print(name_print, "GrlEnv-HalfCheetah-v2")
        elif "_r_" in file_yaml:
            env = Environments('Gym-Reacher-v2')
            steps_p_ep = 500
            print(name_print, "Gym-Reacher-v2")
        elif "_hs_" in file_yaml:
            env = Environments('Gym-HumanoidStandup-v2')
            steps_p_ep = 1000
            print(name_print, "Gym-HumanoidStandup-v2")
        elif "_cr_" in file_yaml:
            env = Environments('Gym-CarRacing-v0')
            steps_p_ep = 1000
            print(name_print, "Gym-CarRacing-v0")
        elif "_humanoid_" in file_yaml:
            env = Environments('Gym-Humanoid-v2')
            steps_p_ep = 1000
            print(name_print, "Gym-Humanoid-v2")
        elif "_ant" in file_yaml:
            env = Environments('Gym-Ant-v2')
            steps_p_ep = 1000
            print(name_print, "Gym-Ant-v2")
        elif "_sw_" in file_yaml:
            env = Environments('Gym-Swimmer-v2')
            steps_p_ep = 1000
            print(name_print, "Gym-Swimmer-v2")
        elif "_wk_" in file_yaml:
            env = Environments('Gym-Walker2d-v2')
            steps_p_ep = 1000
            print(name_print, "Gym-Walker2d-v2")
        else:
            print("no env")
            print(name_print, file_yaml)
            exit(-1)
        return env, steps_p_ep


class IterationMode:
  alternately_persistent = 0
  random = 1
  random_weighted = 2
  online = 3
  random_weighted_by_return = 4
  policy_persistent_random_weighted = 5
  policy_persistent_random_weighted_by_return = 6


class DDPGNetworkConfig:
  def __init__(self, lractor, lrcritic, act1, act2, layer1, layer2, tau, interval):
    # Protected member
    self._lractor = lractor;
    self._lrcritic = lrcritic;
    self._act1 = act1;
    self._act2 = act2;
    self._layer1 = layer1;
    self._layer2 = layer2;
    self._tau = tau;
    self._interval = interval;
    self._gamma = 0;
    self._reward_scale = 0;


  def setGamma(self, gamma):
    self._gamma = gamma


  def setRWScale(self, reward_scale):
    self._reward_scale = reward_scale


  def print(self):
    print("lractor: " + str(self._lractor) + "; lrcritic: " + str(self._lrcritic) + "; act1: " + self._act1 + "; act2: " + self._act2 +
    "; layer1: " + str(self._layer1) + "; layer2: " + str(self._layer2) + "; tau: " + str(self._tau) + "; interval: " + str(self._interval) + "; gamma: " + str(self._gamma) + "; reward_scale: " + str(self._reward_scale))

