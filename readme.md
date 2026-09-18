# RL MLOps Lab — DQN CartPole

A reinforcement learning project implementing a Deep Q-Network (DQN) from scratch using PyTorch and Gymnasium.

The project focuses on understanding the complete reinforcement learning workflow, including neural-network-based Q-learning, exploration and exploitation, experience replay, target networks, model saving, evaluation, and experiment results.

## Project Overview

The agent learns to balance a pole on a moving cart in the CartPole-v1 environment.

The agent observes a 4-dimensional state:

* Cart position
* Cart velocity
* Pole angle
* Pole angular velocity

The agent can choose between two actions:

* Move left
* Move right

The objective is to maximize the total reward by keeping the pole balanced for as long as possible.

## Architecture

```text
CartPole Environment
        ↓
     State
        ↓
   Epsilon-Greedy
        ↓
    DQN Network
        ↓
      Action
        ↓
CartPole Environment
        ↓
  New State + Reward
        ↓
   Replay Buffer
        ↓
 Random Mini-Batch
        ↓
   DQN Training
        ↑
 Target Network
```

## DQN Components

### 1. Q-Network

A neural network maps the 4-dimensional environment state to two Q-values, one for each possible action.

```text
4 state values
     ↓
Linear(4 → 128)
     ↓
ReLU
     ↓
Linear(128 → 128)
     ↓
ReLU
     ↓
Linear(128 → 2)
     ↓
Q-value for action 0
Q-value for action 1
```

### 2. Epsilon-Greedy Exploration

The agent initially explores the environment by selecting random actions.

As training progresses, epsilon decreases and the agent increasingly selects the action with the highest predicted Q-value.

Minimum epsilon:

```text
0.05
```

### 3. Experience Replay

Experiences are stored as:

```text
(state, action, reward, next_state, done)
```

Random mini-batches are sampled from the replay buffer during training.

This reduces the correlation between consecutive experiences and allows previously observed experiences to be reused.

### 4. Target Network

A separate target network is used when calculating future Q-values.

The target network is periodically synchronized with the main Q-network.

This helps stabilize the learning process.

### 5. Model Saving

After training, the learned network parameters are saved to:

```text
models/dqn_cartpole.pth
```

### 6. Evaluation

The saved model is loaded separately and evaluated without epsilon-based exploration.

This separates training performance from actual policy performance.

## Project Structure

```text
rl-mlops-lab/
│
├── src/
│   ├── dqn.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── replay_buffer.py
│
├── models/
│   └── dqn_cartpole.pth
│
├── results/
│   ├── rewards.csv
│   └── reward_curve.png
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Results

The trained model was evaluated over 10 episodes.

```text
Average evaluation reward: 167.2
Best evaluation reward:   218.0
Worst evaluation reward:  133.0
```

The training process also produced a reward history and training-performance graph.

These results demonstrate that the DQN learned meaningful CartPole behavior, although the policy is not perfectly stable.

## What I Learned

This project was developed incrementally rather than using a pre-built DQN implementation.

Key concepts implemented and investigated:

* Q-values
* Bellman targets
* Neural-network function approximation
* Loss calculation
* Backpropagation
* Epsilon-greedy exploration
* Epsilon decay
* Experience replay
* Target networks
* Terminal-state handling
* Model persistence
* Separate model evaluation
* Training metrics
* Git-based experiment history

## Engineering Lessons

A significant debugging issue occurred during development where the terminal-state flag was stored in the replay buffer before it was updated.

The experience was initially stored using the previous value of `done`.

Correct sequence:

```text
Environment step
      ↓
Calculate terminated/truncated
      ↓
Calculate done
      ↓
Store experience
      ↓
Train
```

Fixing this issue substantially improved the learned policy.

This highlighted an important reinforcement-learning engineering principle:

> Correct environment-transition handling is just as important as the neural-network architecture.

## Technologies

* Python
* PyTorch
* Gymnasium
* NumPy
* Matplotlib
* Git

## Future Improvements

The project will be extended with:

* Better DQN hyperparameter configuration
* Multiple training seeds
* More robust evaluation
* Automated tests
* Experiment tracking with MLflow
* CI using GitHub Actions
* Docker-based reproducible environments
* Model/version management
* Deployment workflow
* Monitoring

The long-term goal is to connect these MLOps practices with robotics projects involving ROS, computer vision, and reinforcement learning.
