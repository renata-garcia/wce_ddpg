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
