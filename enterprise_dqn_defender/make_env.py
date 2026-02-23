import gymnasium as gym
import cyberbattle

def make_env():
    env = gym.make("CyberBattleChain-v0")
    return env