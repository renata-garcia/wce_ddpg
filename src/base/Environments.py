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
    else:
      print("Environments wrong id===========================")
      exit(-1)

  def get_env(self):
      return self.env

  def get_obs(self):
    if (self._id == "GrlEnv-Pendulum-v0") or (self._id == "GrlEnv-CartPole-v0"):
      return self._env.observation_space.shape[0] + 1
    elif self._id == "GrlEnv-CartDoublePole-v0":
      return self._env.observation_space.shape[0] + 2
    elif self._id == "GrlEnv-HalfCheetah-v2":
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
    elif self._id == "GrlEnv-HalfCheetah-v2":
      return observation

