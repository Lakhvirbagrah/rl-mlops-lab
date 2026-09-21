# End-to-End Reinforcement Learning MLOps Pipeline — DQN CartPole

End-to-end Reinforcement Learning and MLOps project using PyTorch, MLflow, Docker, FastAPI, GitHub Actions, GHCR, and Render.

The project demonstrates the complete lifecycle of an RL model:

```text
Problem
  ↓
DQN Training
  ↓
MLflow Tracking
  ↓
Model Registry
  ↓
Testing
  ↓
CI/CD
  ↓
Docker + FastAPI
  ↓
Cloud Deployment
  ↓
Monitoring
  ↓
Retraining
  ↓
Champion Promotion
```

## Problem

Train a Deep Q-Network (DQN) agent to solve `CartPole-v1` and then productionize the trained model using an end-to-end MLOps workflow.

The agent observes:

- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

Actions:

- `0` — move left
- `1` — move right

---

## DQN Architecture

```text
State (4)
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
Q(left), Q(right)
```

Implemented:

- Experience replay
- Target network
- Epsilon-greedy exploration
- Bellman targets
- Model saving
- Evaluation
- Reward tracking

---

## MLOps Architecture

```text
DQN Training
   ↓
MLflow Experiment Tracking
   ↓
MLflow Model Registry
   ↓
Candidate / Champion
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
Champion Promotion
```

---

## Model Tracking and Registry

MLflow tracks:

**Parameters**
- learning rate
- epsilon decay
- random seed

**Metrics**
- episode reward
- average reward
- best reward
- loss
- epsilon

Registered model:

```text
DQN-CartPole-Model
```

Aliases:

```text
@candidate
@champion
```

---

## Model Evaluation

Version 1 and Version 2 were evaluated using the same 20 CartPole seeds.

| Model | Average Reward | Best | Worst |
|---|---:|---:|---:|
| Version 1 | 144.65 | 195 | 123 |
| Version 2 | **183.15** | **266** | **146** |

Version 2 passed validation and was promoted to:

```text
@champion
```

---

## Testing and CI/CD

Automated testing uses `pytest`.

Tests cover:

- Q-network output shape
- Epsilon decay
- Minimum epsilon
- Replay buffer sampling

GitHub Actions handles:

```text
Push
  ↓
Install dependencies
  ↓
Run tests
  ↓
Build Docker image
  ↓
Publish to GHCR
```

---

## Docker + FastAPI

The trained model is served using FastAPI inside Docker.

Endpoints:

```text
GET /health
POST /predict
```

Example prediction:

```json
{
  "action": 1,
  "q_left": 9.704,
  "q_right": 9.980,
  "inference_time_seconds": 0.0011
}
```

---

## Cloud Deployment

The Dockerized FastAPI service is deployed on Render.

```text
GitHub
  ↓
GitHub Actions
  ↓
Docker Image
  ↓
GHCR
  ↓
Render
  ↓
Public API
```

---

## Monitoring

API monitoring results:

```text
Successful requests: 20 / 20
Failed requests: 0
Average latency: ~0.281 s
Fastest: ~0.222 s
Slowest: ~0.914 s
Model inference: ~1 ms
```

Model-performance monitoring also evaluates reward against a validation threshold.

Example:

```text
Average reward: 176.0
Threshold: 140

Status: HEALTHY
```

---

## Retraining and Promotion

The project includes a complete retraining loop:

```text
Champion V1
   ↓
Train V2
   ↓
Register V2
   ↓
Candidate
   ↓
Evaluate
   ↓
Compare with Champion
   ↓
Promote V2
   ↓
Redeploy
```

---

## Project Structure

```text
rl-mlops-lab/
├── src/
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── evaluate_registry.py
│   ├── validate_model.py
│   ├── compare_models.py
│   ├── monitor_model.py
│   ├── remote_evaluate.py
│   └── api.py
├── tests/
│   └── test_dqn.py
├── models/
├── results/
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technologies

Python, PyTorch, Gymnasium, MLflow, pytest, GitHub Actions, Docker, FastAPI, Uvicorn, GitHub Container Registry, Render, NumPy, Matplotlib, Git, Linux.

---

## Live Deployment

The trained DQN model is deployed as a FastAPI service on Render.

- Health endpoint: `https://rl-mlops-lab-1.onrender.com/health`
- Prediction endpoint: `https://rl-mlops-lab-1.onrender.com/predict`

## Key Takeaway

This project goes beyond model training and demonstrates the full AI engineering lifecycle:

**train → track → version → test → deploy → monitor → retrain → promote**

GitHub:  
https://github.com/Lakhvirbagrah/rl-mlops-lab