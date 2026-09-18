
import torch
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
import csv

from replay_buffer import (
    create_replay_buffer,
    add_experience,
    sample_experiences
)


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


def choose_action_epsilon_greedy(
    network,
    state,
    epsilon
):

    if random.random() < epsilon:
        return random.randrange(2)

    return choose_action(network, state)


def decay_epsilon(
    epsilon,
    decay_rate=0.995,
    min_epsilon=0.05
):

    epsilon = epsilon * decay_rate

    if epsilon < min_epsilon:
        epsilon = min_epsilon

    return epsilon


def train_from_replay(
    network,
    target_network,
    optimizer,
    replay_buffer,
    batch_size,
    gamma=0.99
):

    experiences = sample_experiences(
        replay_buffer,
        batch_size
    )

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for experience in experiences:

        state, action, reward, next_state, done = experience

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    states = torch.tensor(
        states,
        dtype=torch.float32
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32
    )

    # Current Q-values
    q_values = network(states)

    predicted_q = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # Target Q-values
    with torch.no_grad():

        next_q_values = target_network(
            next_states
        )

        best_future_q = torch.max(
            next_q_values,
            dim=1
        ).values

        target_q = rewards + (
            gamma
            * best_future_q
            * (1 - dones)
        )

    # Loss
    loss = torch.mean(
        (predicted_q - target_q) ** 2
    )

    # Update network
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


if __name__ == "__main__":

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Environment
    env = gym.make("CartPole-v1")

    # Networks
    network = create_q_network()

    target_network = create_q_network()

    target_network.load_state_dict(
        network.state_dict()
    )

    target_network.eval()

    # Optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=0.001
    )

    # Training settings
    epsilon = 1.0

    num_episodes = 100

    batch_size = 32

    replay_capacity = 10000

    replay_buffer = create_replay_buffer(
        replay_capacity
    )

    rewards = []

    # Training
    for episode in range(num_episodes):

        state, info = env.reset()

        total_reward = 0

        done = False

        loss = 0.0

        while not done:

            # Choose action
            action = choose_action_epsilon_greedy(
                network,
                state,
                epsilon
            )

            # Take action
            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            # IMPORTANT:
            # Calculate done BEFORE storing experience
            done = terminated or truncated

            # Store experience
            add_experience(
                replay_buffer,
                state,
                action,
                reward,
                next_state,
                done,
                replay_capacity
            )

            # Train from replay buffer
            if len(replay_buffer) >= batch_size:

                loss = train_from_replay(
                    network,
                    target_network,
                    optimizer,
                    replay_buffer,
                    batch_size,
                    gamma=0.99
                )

            # Move to next state
            state = next_state

            total_reward += reward

        # Update target network every 10 episodes
        if (episode + 1) % 10 == 0:

            target_network.load_state_dict(
                network.state_dict()
            )

        # Reduce exploration
        epsilon = decay_epsilon(
            epsilon
        )

        rewards.append(
            total_reward
        )


        print(
            "Episode:",
            episode + 1,
            "Reward:",
            total_reward,
            "Epsilon:",
            epsilon,
            "Loss:",
            loss
        )

    # Results
    average_reward = (
        sum(rewards) / len(rewards)
    )

    print()
    print("Training Results")
    print("----------------")
    print(
        "Average reward:",
        average_reward
    )

    print(
        "Best reward:",
        max(rewards)
    )

    # Save model
    torch.save(
        network.state_dict(),
        "models/dqn_cartpole.pth"
    )

    print()
    print("Model saved.")
    with open("results/rewards.csv", "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "reward"
        ])

        for episode, reward in enumerate(rewards, start=1):

            writer.writerow([
                episode,
                reward
            ])

    print("Reward history saved.")

    env.close()
