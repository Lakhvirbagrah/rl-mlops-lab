import gymnasium as gym
import torch

from src.dqn import create_q_network


MODEL_PATH = "models/dqn_cartpole.pth"

NUM_EPISODES = 10

MIN_AVERAGE_REWARD = 140


# -----------------------------
# Load model
# -----------------------------

network = create_q_network()

network.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

network.eval()


# -----------------------------
# Evaluate model
# -----------------------------

env = gym.make("CartPole-v1")

rewards = []


for episode in range(NUM_EPISODES):

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


env.close()


# -----------------------------
# Performance summary
# -----------------------------

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
    "Model Performance Monitoring"
)

print(
    "----------------------------"
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

print(
    "Minimum required reward:",
    MIN_AVERAGE_REWARD
)


# -----------------------------
# Retraining trigger
# -----------------------------

if (
    average_reward
    < MIN_AVERAGE_REWARD
):

    print()

    print(
        "PERFORMANCE BELOW THRESHOLD"
    )

    print(
        "Retraining should be triggered."
    )

else:

    print()

    print(
        "MODEL PERFORMANCE HEALTHY"
    )

    print(
        "No retraining required."
    )