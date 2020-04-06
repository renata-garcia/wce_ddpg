
class DDPGNetworkConfig:
  def __init__(self, lractor, lrcritic, act1, act2, layer1, layer2, tau):
    # Protected member
    self._lractor = lractor;
    self._lrcritic = lrcritic;
    self._act1 = act1;
    self._act2 = act2;
    self._layer1 = layer1;
    self._layer2 = layer2;
    self._tau = tau;