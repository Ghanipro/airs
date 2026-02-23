import numpy as np

class Attacker:
    def __init__(self, config, network):
        self.config = config
        self.net = network

    def reset(self):
        pass  # using base_env reset

    def attack(self):
        new_comp = self.net.compromised.copy()

        for i in range(self.net.num_hosts):
            if self.net.compromised[i] == 1 and not self.net.isolated[i]:
                for j in range(self.net.num_hosts):
                    if (self.net.adj[i][j] == 1
                        and not self.net.compromised[j]
                        and not self.net.isolated[j]):

                        prob = self.config.ATTACK_PROB
                        if self.net.patched[j]:
                            prob *= self.config.PATCH_EFFECTIVENESS

                        if np.random.random() < prob:
                            new_comp[j] = 1

        self.net.compromised = new_comp