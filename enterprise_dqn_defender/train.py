import numpy as np
import torch
import config
from dqn_agent import DQNAgent
from make_env import make_env
import matplotlib.pyplot as plt

env = make_env()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

epsilon = config.EPSILON_START
rewards_history = []

for episode in range(config.EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(config.MAX_STEPS):
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(config.EPSILON_MIN, epsilon * config.EPSILON_DECAY)
    rewards_history.append(total_reward)

    print(f"Episode {episode+1} | Reward: {total_reward:.2f}")

torch.save(agent.q_net.state_dict(), config.MODEL_PATH)

plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.show()