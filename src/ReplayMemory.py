import random

"""Memory of all past transitions for experience replay"""

class ReplayMemory():
  def __init__(self):
    self.memory = []

  def sample_minibatch(self, batch_size):
    batch = [self.memory[random.randint(0, len(self.memory) - 1)] for a in range(batch_size)]
    obs = [e[0] for e in batch]
    act = [e[1] for e in batch]
    rew = [e[2] for e in batch]
    nobs = [e[3] for e in batch]
    return obs, act, rew, nobs

  def add(self, obs, act, rew, nobs):
    self.memory.append((obs, act, rew, nobs))

  def size(self):
    return len(self.memory)

