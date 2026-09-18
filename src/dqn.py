import torch
import torch.nn as nn
import gymnasium as gym
import random
from replay_buffer import create_replay_buffer, add_experience
def create_q_network():

    network = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

    return network

def decay_epsilon(epsilon, decay_rate=0.995, min_epsilon=0.05):
    epsilon = epsilon * decay_rate

    if epsilon < min_epsilon:
        epsilon = min_epsilon

    return epsilon


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
def choose_action_epsilon_greedy(network, state, epsilon):

    if random.random() < epsilon:
        return random.randrange(2)

    return choose_action(network, state)

def get_q_value(network, state, action):

    state = torch.tensor(
        state,
        dtype=torch.float32
    )

    state = state.unsqueeze(0)

    q_values = network(state)

    q_value = q_values[0, action]

    return q_value


def calculate_target(network, next_state, reward, done, gamma=0.99):

    next_state = torch.tensor(
        next_state,
        dtype=torch.float32
    )

    next_state = next_state.unsqueeze(0)

    with torch.no_grad():
        next_q_values = network(next_state)

        best_future_q = torch.max(next_q_values)

    if done:
        target = reward
    else:
        target = reward + gamma * best_future_q

    return target

def train_step(
    network,
    optimizer,
    state,
    action,
    reward,
    next_state,
    done,
    gamma=0.99
):

    state = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    next_state = torch.tensor(
        next_state,
        dtype=torch.float32
    ).unsqueeze(0)

    # Q-value for the action we actually took
    q_values = network(state)

    predicted_q = q_values[0, action]

    # Calculate target
    with torch.no_grad():

        next_q_values = network(next_state)

        best_future_q = torch.max(next_q_values)

        if done:
            target_q = torch.tensor(reward)

        else:
            target_q = reward + gamma * best_future_q

    # Calculate loss
    loss = (predicted_q - target_q) ** 2

    # Update network
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()

env = gym.make("CartPole-v1")

network = create_q_network()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

epsilon = 1.0
rewards = []

num_episodes = 100
replay_buffer = create_replay_buffer(10000)

for episode in range(num_episodes):

    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = choose_action_epsilon_greedy(
            network,
            state,
            epsilon
        )

        next_state, reward, terminated, truncated, info = env.step(action)
        add_experience(replay_buffer,state,action,reward,next_state,done,10000)
        done = terminated or truncated

        loss = train_step(
            network,
            optimizer,
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state
        total_reward += reward

    epsilon = decay_epsilon(epsilon)
    rewards.append(total_reward)


    print(
        "Episode:",
        episode + 1,
        "Reward:",
        total_reward,
        "Epsilon:",
        epsilon
    )
average_reward = sum(rewards) / len(rewards)

print("Average reward:", average_reward)
print("Best reward:", max(rewards))
env.close()