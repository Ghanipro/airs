import torch
import numpy as np

from enterprise_env.base_env import EnterpriseAttackEnv
import enterprise_env.config as config
from dqn.model import DQN


def evaluate():

    device = torch.device("cpu")

    env = EnterpriseAttackEnv(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load("dqn_enterprise.pth"))
    model.eval()

    episodes = 50
    total_rewards = []

    for _ in range(episodes):

        state = env.reset()
        done = False
        total = 0

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_t).argmax().item()

            state, reward, done, _ = env.step(action)
            total += reward

        total_rewards.append(total)

    print("Average Reward:", np.mean(total_rewards))


if __name__ == "__main__":
    evaluate()