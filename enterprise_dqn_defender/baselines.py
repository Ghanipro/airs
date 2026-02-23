import numpy as np
from CybORG.Agents import SimpleBlueAgent
from make_env import make_env

def evaluate_random(episodes=50):

    env = make_env()
    rewards = []

    for _ in range(episodes):

        state = env.reset()
        done = False
        total = 0

        while not done:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total += reward

        rewards.append(total)

    return np.mean(rewards), np.std(rewards)


def evaluate_simple_blue(episodes=50):

    env = make_env()
    blue_agent = SimpleBlueAgent()

    rewards = []

    for _ in range(episodes):

        state = env.reset()
        done = False
        total = 0

        while not done:
            action = blue_agent.get_action(state)
            state, reward, done, _ = env.step(action)
            total += reward

        rewards.append(total)

    return np.mean(rewards), np.std(rewards)