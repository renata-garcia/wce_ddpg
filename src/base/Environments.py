import gym
from gym.envs.registration import register
import math

from gym import logger as gymlogger
gymlogger.set_level(40) #error only

#FILE=base/Environments.py; rm $FILE; touch $FILE; chmod 755 $FILE; nano $FILE; ./wce_ddpg.py

class Environments(): #TODO separe files and use design patterns
  def __init__(self, id):
    self._id = id
    self._env = 0
    print("class Environments.py")

    if id == "GrlEnv-Pendulum-v0":
      register(
        id='GrlEnv-Pendulum-v0',
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/pendulum_swingup.yaml"}
      )
      # Create Gym environment
      self._env =  gym.make("GrlEnv-Pendulum-v0")
    elif id == "GrlEnv-CartPole-v0":
      register(
        id='GrlEnv-CartPole-v0',
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/cart_pole.yaml"}
      )
      self._env =  gym.make("GrlEnv-CartPole-v0")
    elif id == "GrlEnv-CartDoublePole-v0":
      register(
        id='GrlEnv-CartDoublePole-v0',
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/cart_double_pole.yaml"}
      )
      self._env =  gym.make("GrlEnv-CartDoublePole-v0")
    elif id == "GrlEnv-HalfCheetah-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/half_cheetah.yaml"}
      )
      self._env =  gym.make("HalfCheetah-v2")
    elif id == "Gym-Reacher-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/reacher.yaml"}
      )
      self._env =  gym.make("Reacher-v2")
    elif id == "Gym-HumanoidStandup-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/humanoid_standup.yaml"}
      )
      print("register")
      self._env =  gym.make("HumanoidStandup-v2")
    elif id == "Gym-CarRacing-v0":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/car_racing.yaml"}
      )
      print("register")
      self._env =  gym.make("CarRacing-v0")
    elif id == "Gym-Humanoid-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/humanoid.yaml"}
      )
      print("register")
      self._env =  gym.make("Humanoid-v2")
    elif id == "Gym-Ant-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/ant.yaml"}
      )
      print("register")
      self._env =  gym.make("Ant-v2")
    elif id == "Gym-Swimmer-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/swimmer.yaml"}
      )
      print("register")
      self._env =  gym.make("Swimmer-v2")
    elif id == "Gym-Walker2d-v2":
      register(
        id=id,
        entry_point='grlenv.grlenv:GrlEnv',
        kwargs={"file": "../cfg/walker.yaml"}
      )
      print("register")
      self._env =  gym.make("Walker2d-v2")
    else:
      print("Environments.py wrong id===========================")
      exit(-1)
    print("id")
    print(id)
    print("..")

  def get_env(self):
      return self.env

  def get_obs(self):
    if (self._id == "GrlEnv-Pendulum-v0") or (self._id == "GrlEnv-CartPole-v0"):
      return self._env.observation_space.shape[0] + 1
    elif self._id == "GrlEnv-CartDoublePole-v0":
      return self._env.observation_space.shape[0] + 2
    else:
      return self._env.observation_space.shape[0]

  def get_obs_trig(self, observation):
    if self._id == "GrlEnv-Pendulum-v0":
      return [math.cos(observation[0]), math.sin(observation[0]), observation[1]]
    elif self._id == "GrlEnv-CartPole-v0":
      # [position of cart, angle of pole, velocity of cart, rotation rate of pole]
      return [observation[0], math.cos(observation[1]), math.sin(observation[1]), observation[2], observation[3]]
    elif self._id == "GrlEnv-CartDoublePole-v0":
      #pos, ang1, ang2, vel, velang1, velang2  #TODO
      return [observation[0], math.cos(observation[1]), math.sin(observation[1]), math.cos(observation[2]), math.sin(observation[2]), observation[3], observation[4], observation[5]]
    else:
      return observation

  def set_reset(self, test):
    if (self._id == "GrlEnv-Pendulum-v0") or (self._id == "GrlEnv-CartPole-v0") or (self._id == "GrlEnv-CartDoublePole-v0"):
      return self._env.reset(test)
    else:
      return self._env.reset()
