import gymnasium as gym
import torch
import mlflow.pytorch

from mlflow import MlflowClient


MODEL_NAME = "DQN-CartPole-Model"
CANDIDATE_ALIAS = "candidate"

MIN_AVERAGE_REWARD = 140

NUM_EPISODES = 10


# --------------------------------
# Load candidate model
# --------------------------------

model_uri = (
    f"models:/{MODEL_NAME}"
    f"@{CANDIDATE_ALIAS}"
)

network = mlflow.pytorch.load_model(
    model_uri
)

network.eval()


# --------------------------------
# Evaluate
# --------------------------------

env = gym.make(
    "CartPole-v1"
)

rewards = []


for episode in range(
    NUM_EPISODES
):

    state, info = env.reset()

    done = False

    total_reward = 0


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


env.close()


# --------------------------------
# Calculate validation metric
# --------------------------------

average_reward = (
    sum(rewards)
    / len(rewards)
)


print()

print(
    "Model Validation"
)

print(
    "----------------"
)

print(
    "Average reward:",
    average_reward
)

print(
    "Required reward:",
    MIN_AVERAGE_REWARD
)


# --------------------------------
# Validation decision
# --------------------------------

if (
    average_reward
    >= MIN_AVERAGE_REWARD
):

    print()

    print(
        "VALIDATION PASSED"
    )


    # Get model version currently
    # assigned to candidate

    client = MlflowClient()


    candidate_version = (
        client
        .get_model_version_by_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS
        )
    )


    version = (
        candidate_version.version
    )


    # Promote candidate to champion

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version
    )


    print(
        "Model promoted:"
    )

    print(
        f"Version {version}"
        " → @champion"
    )


else:

    print()

    print(
        "VALIDATION FAILED"
    )

    print(
        "Candidate was NOT promoted."
    )