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


network = create_q_network()

optimizer = torch.optim.Adam(
    network.parameters(),
    lr=0.001
)

state = [0.1, 0.2, 0.05, -0.1]

action = 1

reward = 1

next_state = [0.2, 0.3, 0.04, -0.08]

done = False

loss = train_step(
    network,
    optimizer,
    state,
    action,
    reward,
    next_state,
    done
)

print("Training loss:", loss)