import csv
import matplotlib.pyplot as plt


episodes = []
rewards = []


with open("results/rewards.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        episodes.append(
            int(row["episode"])
        )

        rewards.append(
            float(row["reward"])
        )


plt.plot(
    episodes,
    rewards
)

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.title(
    "DQN CartPole Training Performance"
)

plt.grid()

plt.savefig(
    "results/reward_curve.png"
)

plt.show()

print("Reward curve saved.")