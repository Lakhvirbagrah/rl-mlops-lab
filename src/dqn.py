import torch
import torch.nn as nn
import gymnasium as gym


def create_q_network():

    network = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

    return network


def choose_action(network, state):

    state = torch.tensor(
        state,
        dtype=torch.float32
    )

    state = state.unsqueeze(0)

    q_values = network(state)

    action = torch.argmax(q_values).item()

    return action


env = gym.make("CartPole-v1")

network = create_q_network()

state, info = env.reset()

total_reward = 0
done = False

while not done:

    action = choose_action(network, state)

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    done = terminated or truncated

print("Episode reward:", total_reward)

env.close()

def get_q_value(network, state, action):

    state = torch.tensor(
        state,
        dtype=torch.float32
    )

    state = state.unsqueeze(0)

    q_values = network(state)

    q_value = q_values[0, action]

    return q_value

network = create_q_network()

state = [0.1, 0.2, 0.05, -0.1]

action = 1

q_value = get_q_value(network, state, action)

print("Action:", action)
print("Q-value for selected action:", q_value.item())