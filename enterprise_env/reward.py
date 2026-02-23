import numpy as np

class Reward:

    def __init__(self, config, network):
        self.config = config
        self.net = network

    def compute(self):
        reward = 0

        for i in range(self.net.num_hosts):
            if self.net.compromised[i]:
                tier = self.net.tiers[i]
                reward -= self.config.COMPROMISE_PENALTY[tier]

        return reward 