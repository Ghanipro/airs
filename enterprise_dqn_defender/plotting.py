import matplotlib.pyplot as plt

def plot_rewards(reward_history):

    plt.figure(figsize=(10,5))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve - EnterpriseScenario")
    plt.grid()
    plt.show()