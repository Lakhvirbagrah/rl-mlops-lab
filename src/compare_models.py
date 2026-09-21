import gymnasium as gym
import torch
import mlflow
import mlflow.pytorch

from mlflow import MlflowClient


MODEL_NAME = "DQN-CartPole-Model"

V1_URI = f"models:/{MODEL_NAME}/1"
V2_URI = f"models:/{MODEL_NAME}/2"

NUM_EPISODES = 20
MIN_AVERAGE_REWARD = 140


# Make sure evaluation runs go into the same MLflow experiment
mlflow.set_experiment("DQN-CartPole")


def evaluate_model(model_uri):
    network = mlflow.pytorch.load_model(model_uri)
    network.eval()

    env = gym.make("CartPole-v1")

    rewards = []

    # Same seeds for both models
    for episode in range(NUM_EPISODES):
        state, info = env.reset(seed=1000 + episode)

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = network(state_tensor)

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
            ) = env.step(action)

            done = terminated or truncated

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    return {
        "average": sum(rewards) / len(rewards),
        "best": max(rewards),
        "worst": min(rewards)
    }


# -----------------------------
# Evaluate Version 1
# -----------------------------

print("Evaluating Version 1...")

v1_results = evaluate_model(V1_URI)


# -----------------------------
# Evaluate Version 2
# -----------------------------

print("Evaluating Version 2...")

v2_results = evaluate_model(V2_URI)


# -----------------------------
# Print comparison
# -----------------------------

print()
print("Champion vs Candidate")
print("---------------------")

print()
print("Version 1")
print("Average:", v1_results["average"])
print("Best:", v1_results["best"])
print("Worst:", v1_results["worst"])

print()
print("Version 2")
print("Average:", v2_results["average"])
print("Best:", v2_results["best"])
print("Worst:", v2_results["worst"])


# -----------------------------
# Log controlled evaluation to MLflow
# -----------------------------

with mlflow.start_run(
    run_name="V1-Controlled-Evaluation"
):
    mlflow.log_param(
        "model_version",
        1
    )

    mlflow.log_param(
        "evaluation_episodes",
        NUM_EPISODES
    )

    mlflow.log_param(
        "evaluation_seed_start",
        1000
    )

    mlflow.log_metric(
        "evaluation_average_reward",
        v1_results["average"]
    )

    mlflow.log_metric(
        "evaluation_best_reward",
        v1_results["best"]
    )

    mlflow.log_metric(
        "evaluation_worst_reward",
        v1_results["worst"]
    )


with mlflow.start_run(
    run_name="V2-Controlled-Evaluation"
):
    mlflow.log_param(
        "model_version",
        2
    )

    mlflow.log_param(
        "evaluation_episodes",
        NUM_EPISODES
    )

    mlflow.log_param(
        "evaluation_seed_start",
        1000
    )

    mlflow.log_metric(
        "evaluation_average_reward",
        v2_results["average"]
    )

    mlflow.log_metric(
        "evaluation_best_reward",
        v2_results["best"]
    )

    mlflow.log_metric(
        "evaluation_worst_reward",
        v2_results["worst"]
    )


# -----------------------------
# Promotion decision
# -----------------------------

if (
    v2_results["average"] > v1_results["average"]
    and
    v2_results["average"] >= MIN_AVERAGE_REWARD
):
    print()
    print("VERSION 2 PASSED")
    print("Promoting V2 to champion...")

    client = MlflowClient()

    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        "2"
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        "2"
    )

    print()
    print("Version 2 is now @champion")

else:
    print()
    print("VERSION 2 NOT PROMOTED")
    print("Version 1 remains champion.")