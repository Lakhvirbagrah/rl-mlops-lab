import gymnasium as gym
import torch
import mlflow.pytorch


# Load model from MLflow Registry using alias
model_uri = "models:/DQN-CartPole-Model@candidate"

network = mlflow.pytorch.load_model(
    model_uri
)

network.eval()


# Environment
env = gym.make(
    "CartPole-v1"
)

num_episodes = 10

rewards = []


for episode in range(
    num_episodes
):

    state, info = env.reset()

    total_reward = 0

    done = False


    while not done:

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1
        ).item()


        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        done = (
            terminated
            or truncated
        )

        state = next_state

        total_reward += reward


    rewards.append(
        total_reward
    )

    print(
        "Episode:",
        episode + 1,
        "Reward:",
        total_reward
    )


average_reward = (
    sum(rewards)
    / len(rewards)
)

best_reward = max(
    rewards
)

worst_reward = min(
    rewards
)


print()

print(
    "Registry Model Evaluation"
)

print(
    "-------------------------"
)

print(
    "Model URI:",
    model_uri
)

print(
    "Average reward:",
    average_reward
)

print(
    "Best reward:",
    best_reward
)

print(
    "Worst reward:",
    worst_reward
)


env.close()
