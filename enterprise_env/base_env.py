import numpy as np
import gym
from gym import spaces
from enterprise_env.network import EnterpriseNetwork
from enterprise_env.attacker import Attacker
from enterprise_env.defender import Defender
from enterprise_env.reward import Reward

class EnterpriseAttackEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()

        self.net = EnterpriseNetwork(config)
        self.attacker = Attacker(config, self.net)
        self.defender = Defender(config, self.net)
        self.rewarder = Reward(config, self.net)

        self.max_steps = config.MAX_STEPS
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.net.state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.defender.total_actions)

    def reset(self):
        self.net.reset()
        self.attacker.reset()
        self.defender.reset()
        self.current_step = 0
        return self.net.get_state()

    def step(self, action):
        self.defender.apply_action(action)
        self.attacker.attack()

        reward = self.rewarder.compute()
        self.current_step += 1

        done = (self.current_step >= self.max_steps) or self.net.is_terminal()

        return self.net.get_state(), reward, done, {}

    def render(self, mode="human"):
        # optional for debugging
        print("Step:", self.current_step)
        print("Compromised:", self.net.compromised)