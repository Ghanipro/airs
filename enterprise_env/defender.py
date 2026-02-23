class Defender:

    def __init__(self, config, network):
        self.config = config
        self.net = network
        self.total_actions = 1 + 4 * self.net.num_hosts

    def reset(self):
        pass

    def apply_action(self, action):
        # 0 = do nothing
        if action == 0:
            return

        host = (action - 1) // 4
        act_type = (action - 1) % 4

        if act_type == 0:
            self.net.patched[host] = 1
        elif act_type == 1:
            self.net.isolated[host] = 1
        elif act_type == 2:
            self.net.compromised[host] = 0
        elif act_type == 3:
            self.net.privilege[host] += 1