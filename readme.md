# RL MLOps Lab — End-to-End DQN CartPole Pipeline

An end-to-end Reinforcement Learning and MLOps project implementing a Deep Q-Network (DQN) from scratch using PyTorch and Gymnasium, then taking the trained model through experiment tracking, model registry, testing, CI/CD, containerization, API serving, cloud deployment, monitoring, retraining, and controlled model promotion.

The goal of this project is not only to train an RL agent, but to demonstrate the complete engineering lifecycle of a machine-learning system.

---

## Project Workflow

```text
Problem Definition
        ↓
DQN Architecture
        ↓
Model Training
        ↓
MLflow Experiment Tracking
        ↓
Model Registry
        ↓
Model Validation
        ↓
Automated Testing
        ↓
Continuous Integration
        ↓
Docker Containerization
        ↓
FastAPI Model Serving
        ↓
Continuous Delivery
        ↓
Cloud Deployment
        ↓
Service Monitoring
        ↓
Model Performance Monitoring
        ↓
Retraining
        ↓
Candidate vs Champion Evaluation
        ↓
Champion Promotion
        ↓
Redeployment
        ↓
Monitoring Again
```

---

# 1. Problem

The agent learns to balance a pole on a moving cart in the `CartPole-v1` environment.

The environment provides a four-dimensional state:

- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

The agent can choose between two actions:

- `0` — move the cart left
- `1` — move the cart right

The objective is to maximize cumulative reward by keeping the pole balanced for as long as possible.

This relatively simple environment was selected so that the project could focus not only on reinforcement-learning theory, but also on the complete ML engineering lifecycle surrounding the model.

---

# 2. System Architecture

```text
                       ┌─────────────────────┐
                       │   CartPole-v1 Env   │
                       └──────────┬──────────┘
                                  │
                                  │ State
                                  ▼
                       ┌─────────────────────┐
                       │   Epsilon-Greedy    │
                       │  Action Selection   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │     DQN Network     │
                       │      PyTorch        │
                       └──────────┬──────────┘
                                  │
                                  │ Action
                                  ▼
                       ┌─────────────────────┐
                       │   CartPole-v1 Env   │
                       └──────────┬──────────┘
                                  │
                       State / Reward / Done
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   Replay Buffer     │
                       └──────────┬──────────┘
                                  │
                           Random Batch
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    DQN Training     │
                       └──────────┬──────────┘
                                  ▲
                                  │
                       ┌──────────┴──────────┐
                       │   Target Network    │
                       └─────────────────────┘
```

The trained model then moves through the MLOps pipeline:

```text
DQN Training
    ↓
MLflow Tracking
    ↓
Model Registry
    ↓
Validation
    ↓
pytest
    ↓
GitHub Actions CI
    ↓
Docker
    ↓
FastAPI
    ↓
GitHub Actions CD
    ↓
GitHub Container Registry
    ↓
Render Deployment
    ↓
Monitoring
    ↓
Retraining
    ↓
Candidate vs Champion
    ↓
Champion Promotion
```

---

# 3. DQN Architecture

The Q-network maps the four CartPole state values to two Q-values.

```text
4 State Values
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
Q(Action Left)
Q(Action Right)
```

The action with the highest Q-value is selected during greedy inference.

---

# 4. Reinforcement Learning Components

## Q-Network

A PyTorch neural network is used as a function approximator for the action-value function.

Instead of storing Q-values in a table, the neural network predicts Q-values from the continuous CartPole state.

---

## Epsilon-Greedy Exploration

During training, the agent balances exploration and exploitation.

```text
Random number < epsilon
        ↓
Random Action

Otherwise
        ↓
Action with highest Q-value
```

Epsilon decreases over time so the agent gradually shifts from exploration toward exploitation.

Minimum epsilon:

```text
0.05
```

---

## Experience Replay

Every interaction is stored as:

```text
(state, action, reward, next_state, done)
```

Experiences are stored in a replay buffer.

Random mini-batches are sampled during training.

Benefits:

- Reduces correlation between consecutive experiences
- Reuses previously observed transitions
- Improves training stability
- Makes learning more sample-efficient

---

## Target Network

A separate target network is used to calculate future Q-values.

```text
Main Network
     ↓
Learns continuously

Target Network
     ↓
Updated periodically
```

Using a target network prevents the Bellman target from changing too rapidly during optimization.

---

## Bellman Target

For non-terminal states:

```text
Target =
Reward
+
Gamma × Maximum Future Q-value
```

For terminal states:

```text
Target = Reward
```

Correct terminal-state handling is essential for stable DQN learning.

---

# 5. Important RL Bug Fix

During development, a terminal-state handling issue was discovered.

The replay buffer was initially storing the previous value of `done` instead of the value calculated from the latest environment step.

Incorrect sequence:

```text
Store experience
      ↓
Calculate done
```

Correct sequence:

```text
Environment Step
      ↓
terminated / truncated
      ↓
Calculate done
      ↓
Store Experience
      ↓
Train
```

Correcting this issue significantly improved policy performance.

This demonstrated an important RL engineering lesson:

> Correct environment-transition logic can be just as important as neural-network architecture.

---

# 6. Training

Training is implemented using PyTorch.

The training pipeline includes:

- Q-network creation
- Target-network synchronization
- Replay-buffer sampling
- Epsilon-greedy exploration
- Bellman target calculation
- Mean squared error loss
- Backpropagation
- Optimizer updates
- Epsilon decay
- Reward logging
- Model saving

The trained PyTorch parameters are saved to:

```text
models/dqn_cartpole.pth
```

---

# 7. MLflow Experiment Tracking

MLflow is used to track DQN experiments.

Each training run records configuration, metrics, and model artifacts.

## Parameters

Examples include:

```text
learning_rate
epsilon_decay
random_seed
```

## Metrics

Examples include:

```text
episode_reward
average_reward
best_reward
loss
epsilon
```

## Artifacts

Examples include:

```text
trained model
reward history
training results
```

This makes training runs reproducible and allows different hyperparameter configurations to be compared.

---

# 8. Experiment Results

Several DQN configurations were evaluated during development.

One important experiment changed epsilon decay from:

```text
0.995
```

to:

```text
0.99
```

and produced stronger learning performance in that run.

Because individual RL runs can vary due to randomness, training metrics are not used alone for production model promotion.

Registered models are evaluated separately under identical environment conditions.

---

# 9. MLflow Model Registry

The trained model is stored in the MLflow Model Registry.

Registered model:

```text
DQN-CartPole-Model
```

The registry is used for:

- Model versioning
- Model lineage
- Candidate management
- Champion management
- Controlled promotion
- Model rollback capability

Two aliases are used:

```text
@candidate
@champion
```

A newly trained model is treated as a candidate until it passes evaluation.

---

# 10. Model Validation

A validation script evaluates the candidate model before promotion.

The project uses a minimum average reward threshold:

```text
140
```

This threshold is a project validation rule used for this learning pipeline rather than a universal CartPole production standard.

The validation workflow is:

```text
Candidate Model
      ↓
Evaluate
      ↓
Average Reward >= Threshold?
      ↓
Yes → Eligible for Promotion
No  → Reject Candidate
```

---

# 11. Automated Testing

Software tests are implemented using `pytest`.

Current tests validate:

- Q-network output shape
- Epsilon decay
- Minimum epsilon behavior
- Replay-buffer sampling

Run tests using:

```bash
pytest
```

The purpose of these tests is different from model evaluation.

```text
Software Testing
        ↓
Is the code behaving correctly?

Model Evaluation
        ↓
Is the trained policy performing well?
```

Both are required in a production-oriented ML workflow.

---

# 12. Continuous Integration

GitHub Actions is used for Continuous Integration.

The CI workflow automatically:

```text
Git Push / Pull Request
        ↓
Checkout Repository
        ↓
Set Up Python
        ↓
Install Dependencies
        ↓
Run pytest
        ↓
Pass / Fail
```

This prevents broken software changes from progressing through the deployment pipeline.

---

# 13. Docker Containerization

The inference application is containerized using Docker.

The Docker image contains:

- Python runtime
- Project source code
- Python dependencies
- DQN model
- FastAPI application

Build the container locally:

```bash
docker build -t rl-mlops-lab .
```

Docker provides a reproducible environment so the same application can run consistently across machines and cloud environments.

---

# 14. FastAPI Model Serving

The trained DQN model is exposed through a REST API using FastAPI.

The API loads the trained model and performs inference without modifying model weights.

## Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "DQN-CartPole",
  "model_loaded": true
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Example input:

```json
{
  "cart_position": 0,
  "cart_velocity": 0,
  "pole_angle": 0.03,
  "pole_angular_velocity": 0
}
```

Example response from the deployed service:

```json
{
  "action": 1,
  "q_left": 9.70428466796875,
  "q_right": 9.980449676513672,
  "inference_time_seconds": 0.0011241436004638672
}
```

The API returns:

- Selected action
- Left-action Q-value
- Right-action Q-value
- Neural-network inference time

---

# 15. Continuous Delivery

A GitHub Actions CD pipeline builds the Docker application and publishes the container image.

```text
Git Push
    ↓
GitHub Actions
    ↓
Build Docker Image
    ↓
Publish Image
    ↓
GitHub Container Registry
```

Container images are stored in GitHub Container Registry (GHCR).

This separates application source code from deployable application artifacts.

---

# 16. Cloud Deployment

The containerized FastAPI inference service is deployed using Render.

Deployment architecture:

```text
GitHub Repository
       ↓
GitHub Actions
       ↓
Docker Image
       ↓
GitHub Container Registry
       ↓
Render
       ↓
Running FastAPI Service
       ↓
Public REST API
```

The deployed application uses the cloud-provided `PORT` environment variable with a local default of port `8000`.

---

# 17. Service Monitoring

The deployed API is monitored independently from model performance.

Service monitoring measures:

- Request success
- Request failure
- End-to-end latency
- Model inference time
- API health

A monitoring test sent 20 requests to the deployed application.

Results:

| Metric | Result |
|---|---:|
| Successful requests | 20 / 20 |
| Failed requests | 0 |
| Average end-to-end latency | ~0.281 s |
| Fastest request | ~0.222 s |
| Slowest request | ~0.914 s |
| Model inference time | ~1 ms |

This demonstrates the distinction between model execution time and complete network/API latency.

```text
Model Inference
≈ 1 ms

Total API Request
≈ hundreds of milliseconds
```

This is particularly important for robotics systems where real-time control decisions should generally happen locally rather than through a remote cloud API.

---

# 18. Server-Side Logging

The FastAPI application records inference information including:

```text
Selected Action
Q_left
Q_right
Inference Time
```

Example concept:

```text
Prediction
Action = 1
Q_left = 9.704
Q_right = 9.980
Inference_time ≈ 0.0011 seconds
```

This provides basic operational visibility into the deployed inference service.

---

# 19. Model Performance Monitoring

Service health does not guarantee that the ML model is still performing well.

A separate model-monitoring script evaluates the DQN policy.

Example monitoring result:

```text
Average reward: 176.0
Best reward:    211.0
Worst reward:   150.0
Threshold:      140
```

Result:

```text
MODEL PERFORMANCE HEALTHY
No retraining required
```

The project therefore separates two monitoring concerns:

```text
Service Monitoring
        ↓
Is the API available and responsive?

Model Monitoring
        ↓
Is the model still producing acceptable performance?
```

---

# 20. Retraining

To demonstrate the complete lifecycle, a new DQN model was intentionally retrained using a different random seed.

The new model was registered as:

```text
Version 2
```

instead of replacing Version 1 directly.

This creates a controlled model lifecycle:

```text
Champion V1
      ↓
Train New Model
      ↓
Register V2
      ↓
Candidate
      ↓
Evaluate
      ↓
Promote or Reject
```

---

# 21. Champion vs Candidate Evaluation

Version 1 and Version 2 were evaluated under identical conditions.

Both models were tested using the same 20 CartPole environment seeds.

This reduces variability caused by evaluating models on different random starting conditions.

## Registered Model Evaluation Results

| Model | Average Reward | Best Reward | Worst Reward |
|---|---:|---:|---:|
| Version 1 | 144.65 | 195 | 123 |
| Version 2 | **183.15** | **266** | **146** |

Version 2 improved:

- Average reward
- Best reward
- Worst reward

and exceeded the validation threshold.

---

# 22. Champion Promotion

After successful evaluation, Version 2 was promoted to:

```text
@champion
```

The final promotion flow:

```text
Version 1 Champion
        ↓
Train Version 2
        ↓
Register Version 2
        ↓
Assign Candidate
        ↓
Evaluate V1 and V2
Using Same Seeds
        ↓
Compare Performance
        ↓
Validation Threshold Passed
        ↓
Promote V2
        ↓
V2 = @champion
```

The updated model was then redeployed and the public prediction API was verified successfully.

---

# 23. Complete MLOps Lifecycle

The final system demonstrates the complete machine-learning lifecycle:

```text
Problem
   ↓
Architecture
   ↓
DQN Training
   ↓
MLflow Tracking
   ↓
Model Registry
   ↓
Validation
   ↓
Automated Testing
   ↓
CI
   ↓
Docker
   ↓
FastAPI
   ↓
CD
   ↓
Cloud Deployment
   ↓
Service Monitoring
   ↓
Model Monitoring
   ↓
Retraining
   ↓
Candidate Evaluation
   ↓
Champion Promotion
   ↓
Redeployment
   ↓
Monitoring Again
```

---

# 24. Project Structure

```text
rl-mlops-lab/
│
├── src/
│   ├── __init__.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── evaluate.py
│   ├── evaluate_registry.py
│   ├── validate_model.py
│   ├── compare_models.py
│   ├── monitor_model.py
│   ├── remote_evaluate.py
│   ├── api.py
│   └── plot_results.py
│
├── tests/
│   └── test_dqn.py
│
├── models/
│   └── dqn_cartpole.pth
│
├── results/
│   ├── rewards.csv
│   └── reward_curve.png
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── .gitignore
```

Adjust this section if the exact repository files differ.

---

# 25. Technologies

## Machine Learning

- Python
- PyTorch
- Gymnasium
- NumPy

## Reinforcement Learning

- Deep Q-Network
- Q-Learning
- Bellman Targets
- Experience Replay
- Target Networks
- Epsilon-Greedy Exploration

## MLOps

- MLflow
- Experiment Tracking
- Model Registry
- Model Versioning
- Candidate / Champion Promotion
- Model Validation
- Model Monitoring
- Retraining Workflow

## Software Engineering

- Git
- GitHub
- pytest
- Linux
- Conda

## Deployment

- Docker
- FastAPI
- Uvicorn
- REST APIs
- GitHub Actions
- GitHub Container Registry
- Render

## Visualization

- Matplotlib

---

# 26. Key Engineering Lessons

This project demonstrated several important AI engineering principles.

### Training a model is only one part of the lifecycle

A usable ML system also requires:

```text
Tracking
Testing
Versioning
Deployment
Monitoring
Retraining
```

### Model performance and software reliability are different

A model can perform well while the API is broken.

An API can be healthy while the model performs poorly.

Both need independent monitoring.

### Training metrics should not automatically promote a model

A newly trained model must be evaluated independently before replacing the current champion.

### Reproducible comparisons matter

Using identical evaluation seeds provides a more meaningful comparison between model versions.

### Real-time inference and cloud inference have different requirements

The neural network can execute in approximately milliseconds, while internet/API latency can be hundreds of milliseconds.

For real-time robotics control, inference should normally run close to the robot.

---

# 27. Original Model Evaluation

Before the MLOps workflow was added, the saved DQN model was evaluated separately over 10 episodes.

```text
Average reward: 167.2
Best reward:    218.0
Worst reward:   133.0
```

This early evaluation confirmed that the DQN implementation had learned meaningful CartPole behavior.

The later Model Registry workflow introduced controlled versioning and identical-seed comparison for production-style model evaluation.

---

# 28. Future Work

The next stage is to transfer the same engineering principles to robotics.

Planned architecture:

```text
ROS / Gazebo
      +
YOLOv5 Perception
      +
DQN Control
      +
MLflow
      +
Testing
      +
CI/CD
      +
Monitoring
```

The target system is a TurtleBot simulation that detects a randomly positioned object using YOLOv5 and learns to center and navigate toward the object using reinforcement learning.

Future areas include:

- Headless Gazebo training
- ROS-based RL environment design
- YOLO + DQN integration
- Robotics experiment tracking
- Multi-metric robot validation
- Simulation testing
- Model version compatibility
- Robot-side inference
- Sim-to-real considerations

---

# 29. Repository

GitHub:

```text
https://github.com/Lakhvirbagrah/rl-mlops-lab
```

---

# Author

**Lakhvir Singh**

AI/ML Engineer focused on:

- Reinforcement Learning
- MLOps
- Robotics
- Computer Vision
- Machine Learning Deployment