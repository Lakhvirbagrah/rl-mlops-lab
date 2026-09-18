import gymnasium as gym
import torch

from dqn import create_q_network, choose_action


env = gym.make("CartPole-v1")

network = create_q_network()

network.load_state_dict(
    torch.load(
        "models/dqn_cartpole.pth",
        map_location="cpu"
    )
)

network.eval()

num_episodes = 10
rewards = []


for episode in range(num_episodes):

    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = choose_action(network, state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    print(
        "Episode:",
        episode + 1,
        "Reward:",
        total_reward
    )


average_reward = sum(rewards) / len(rewards)

print()
print("Evaluation Results")
print("------------------")
print("Average reward:", average_reward)
print("Best reward:", max(rewards))
print("Worst reward:", min(rewards))


env.close()