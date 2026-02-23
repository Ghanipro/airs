import torch
import config
from dqn_agent import DQNAgent
from make_env import make_env
import numpy as np

env = make_env()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load(config.MODEL_PATH))

episodes = 50
total_rewards = []

for _ in range(episodes):
    state = env.reset()
    episode_reward = 0

    for _ in range(config.MAX_STEPS):
        action = agent.select_action(state, epsilon=0)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break

    total_rewards.append(episode_reward)

print("Average Reward:", np.mean(total_rewards))