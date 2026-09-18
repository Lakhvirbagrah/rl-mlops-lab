import gymnasium as gym
import numpy as np


env = gym.make("CartPole-v1")

num_episodes = 100

rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    rewards.append(total_reward)

    print(f"Episode {episode + 1}: Reward = {total_reward}")


print("\nTraining finished")

print("Average reward:",
      np.mean(rewards))

env.close()