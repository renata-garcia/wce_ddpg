
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