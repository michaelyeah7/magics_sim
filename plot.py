import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

def plot_pendulum():
    file = "examples/models/pendulum/" + "pendulum_rewards_episode_200_" + ".txt"
    with open(file, "rb") as f:
        data = pickle.load(f)

    x1 = data
    x1 = np.asarray(data).reshape(-1,10).T
    time = np.asarray(range(x1.shape[1])) * 10

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="pendulum reward")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Pendulum")
    plt.savefig('examples/models/pendulum/pendulum_svg_agent_value_loss_episode_%d_' % 200 + '.png')
    plt.close()

def plot_cartpole():
    file = "examples/models/cartpole/" + "cartpole_rewards_episode_1210_" + ".txt"
    with open(file, "rb") as f:
        data = pickle.load(f)

    x1 = data[:1200]
    x1 = np.asarray(x1).reshape(-1,50).T
    time = np.asarray(range(x1.shape[1])) * 50

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="cartpole reward")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Cartpole Rewards")
    plt.savefig('examples/models/cartpole/cartpole_svg_agent_value_loss_episode_%d_' % 1210 + '.png')
    plt.close()

def plot_arm():
    file = "examples/models/arm/" + "arm_rewards_episode_230_" + ".txt"
    with open(file, "rb") as f:
        data = pickle.load(f)

    x1 = data
    x1 = np.asarray(data).reshape(-1,10).T
    time = np.asarray(range(x1.shape[1])) * 10

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="arm reward")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Arm Rewards")
    plt.savefig('examples/models/arm/arm_svg_agent_value_loss_episode_%d_' % 230 + '.png')
    plt.close()

if __name__ == '__main__':
    # plot_pendulum()
    plot_cartpole()
    # plot_arm()