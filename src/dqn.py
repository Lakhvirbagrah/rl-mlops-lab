import torch
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
import csv

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

from src.replay_buffer import (
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

    action = torch.argmax(
        q_values
    ).item()

    return action


def choose_action_epsilon_greedy(
    network,
    state,
    epsilon
):

    if random.random() < epsilon:

        return random.randrange(2)

    return choose_action(
        network,
        state
    )


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

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


if __name__ == "__main__":

    # ---------------------------
    # Hyperparameters
    # ---------------------------

    learning_rate = 0.001

    gamma = 0.99

    epsilon_start = 1.0

    epsilon_decay = 0.99

    min_epsilon = 0.05

    batch_size = 32

    replay_capacity = 10000

    num_episodes = 100

    target_update_frequency = 10

    seed = 123

    # Working epsilon
    epsilon = epsilon_start


    # ---------------------------
    # Reproducibility
    # ---------------------------

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


    # ---------------------------
    # MLflow experiment
    # ---------------------------

    mlflow.set_experiment(
        "DQN-CartPole"
    )


    # ---------------------------
    # Environment
    # ---------------------------

    env = gym.make(
        "CartPole-v1"
    )


    # ---------------------------
    # Networks
    # ---------------------------

    network = create_q_network()

    target_network = create_q_network()

    target_network.load_state_dict(
        network.state_dict()
    )

    target_network.eval()


    # ---------------------------
    # Optimizer
    # ---------------------------

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate
    )


    # ---------------------------
    # Replay Buffer
    # ---------------------------

    replay_buffer = create_replay_buffer(
        replay_capacity
    )

    rewards = []


    # ==========================================
    # MLflow Run
    # ==========================================

    with mlflow.start_run(
        run_name="DQN-Retrained-V2"
    ):

        # ---------------------------
        # Log parameters
        # ---------------------------

        mlflow.log_params({

            "learning_rate":
                learning_rate,

            "gamma":
                gamma,

            "epsilon_start":
                epsilon_start,

            "epsilon_decay":
                epsilon_decay,

            "min_epsilon":
                min_epsilon,

            "batch_size":
                batch_size,

            "replay_capacity":
                replay_capacity,

            "episodes":
                num_episodes,

            "target_update_frequency":
                target_update_frequency,

            "random_seed":
                seed

        })


        # ---------------------------
        # Training
        # ---------------------------

        for episode in range(
            num_episodes
        ):

            state, info = env.reset()

            total_reward = 0

            done = False

            loss = 0.0


            while not done:

                # Choose action
                action = (
                    choose_action_epsilon_greedy(
                        network,
                        state,
                        epsilon
                    )
                )


                # Take action
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(
                    action
                )


                # Calculate done first
                done = (
                    terminated
                    or truncated
                )


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


                # Train from replay
                if (
                    len(replay_buffer)
                    >= batch_size
                ):

                    loss = (
                        train_from_replay(
                            network,
                            target_network,
                            optimizer,
                            replay_buffer,
                            batch_size,
                            gamma=gamma
                        )
                    )


                state = next_state

                total_reward += reward


            # ---------------------------
            # Update target network
            # ---------------------------

            if (
                (episode + 1)
                % target_update_frequency
                == 0
            ):

                target_network.load_state_dict(
                    network.state_dict()
                )


            # ---------------------------
            # Reduce exploration
            # ---------------------------

            epsilon = decay_epsilon(
                epsilon,
                decay_rate=epsilon_decay,
                min_epsilon=min_epsilon
            )


            rewards.append(
                total_reward
            )


            # ---------------------------
            # Log episode metrics
            # ---------------------------

            mlflow.log_metric(
                "episode_reward",
                total_reward,
                step=episode + 1
            )

            mlflow.log_metric(
                "loss",
                loss,
                step=episode + 1
            )

            mlflow.log_metric(
                "epsilon",
                epsilon,
                step=episode + 1
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


        # ---------------------------
        # Final results
        # ---------------------------

        average_reward = (
            sum(rewards)
            / len(rewards)
        )

        best_reward = max(
            rewards
        )


        mlflow.log_metric(
            "average_reward",
            average_reward
        )

        mlflow.log_metric(
            "best_reward",
            best_reward
        )


        print()

        print(
            "Training Results"
        )

        print(
            "----------------"
        )

        print(
            "Average reward:",
            average_reward
        )

        print(
            "Best reward:",
            best_reward
        )


        # ---------------------------
        # Save PyTorch model file
        # ---------------------------

        torch.save(
            network.state_dict(),
            "models/dqn_cartpole.pth"
        )

        print()

        print(
            "Model saved."
        )


        # Log normal model artifact
        mlflow.log_artifact(
            "models/dqn_cartpole.pth"
        )


        # ---------------------------
        # Save reward CSV
        # ---------------------------

        with open(
            "results/rewards.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                "episode",
                "reward"
            ])

            for (
                episode,
                reward
            ) in enumerate(
                rewards,
                start=1
            ):

                writer.writerow([
                    episode,
                    reward
                ])


        print(
            "Reward history saved."
        )


        mlflow.log_artifact(
            "results/rewards.csv"
        )


        # ==========================================
        # MLflow Model Registry
        # ==========================================

        # Example CartPole state:
        # [cart position,
        #  cart velocity,
        #  pole angle,
        #  pole angular velocity]



        input_example = np.array(
            [[0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32
        )

        mlflow.pytorch.log_model(
            pytorch_model=network,
            name="dqn-model",
            registered_model_name="DQN-CartPole-Model",
            input_example=input_example,
            serialization_format="pickle"
        )
        client = MlflowClient()

        client.set_registered_model_alias(
            "DQN-CartPole-Model",
            "candidate",
            "1"
        )

        print("Alias 'candidate' assigned to model version 1.")
        print("Model registered with MLflow.")


    env.close()