import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from enterprise_env.base_env import EnterpriseAttackEnv
import enterprise_env.config as config
from dqn.agent import DQNAgent


def train():

    device = torch.device("cpu")

    env = EnterpriseAttackEnv(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device)

    episodes = 500
    rewards_history = []

    for ep in tqdm(range(episodes)):

        state = env.reset()
        total_reward = 0
        done = False

        while not done:

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        print(f"Episode {ep} | Reward: {total_reward:.2f}")

    torch.save(agent.policy_net.state_dict(), "dqn_enterprise.pth")

    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Performance")
    plt.savefig("training_curve.png")
    plt.show()


if __name__ == "__main__":
    train()