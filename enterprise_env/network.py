import numpy as np

class EnterpriseNetwork:

    def __init__(self, config):
        self.cfg = config

        self.num_hosts = sum(config.HOST_COUNTS)
        self.tiers = (
            [0] * config.HOST_COUNTS[0]
            + [1] * config.HOST_COUNTS[1]
            + [2] * config.HOST_COUNTS[2]
            + [3] * config.HOST_COUNTS[3]
        )

        self.adj = self.build_adj()
        self.state_dim = (
            self.num_hosts * (1 + 1 + 1 + 1)
        )  # compromised + privilege + patched + isolated

    def build_adj(self):
        adj = np.zeros((self.num_hosts, self.num_hosts))
        for i in range(self.num_hosts):
            for j in range(self.num_hosts):
                if self.tiers[j] == self.tiers[i] + 1:
                    adj[i][j] = 1
        return adj

    def reset(self):
        self.compromised = np.zeros(self.num_hosts)
        self.privilege = np.zeros(self.num_hosts)
        self.patched = np.zeros(self.num_hosts)
        self.isolated = np.zeros(self.num_hosts)

        # initial DMZ breach
        start = np.random.choice(np.where(np.array(self.tiers) == 0)[0])
        self.compromised[start] = 1

    def get_state(self):
        return np.concatenate(
            [
                self.compromised,
                self.privilege,
                self.patched,
                self.isolated,
            ]
        ).astype(np.float32)

    def is_terminal(self):
        # terminal if critical compromised
        return any(self.compromised[i] == 1 for i in self.cfg.CRITICAL_HOSTS)