import gym
from gym.envs.registration import register

from gym import logger as gymlogger
gymlogger.set_level(40) #error only

class Environments():
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
    else:
      print("Environments wrong id===========================")
      exit(-1)

  def get_env(self):
      return self.env

  def get_obs(self):
      return self._env.observation_space.shape[0]