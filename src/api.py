from fastapi import FastAPI
from pydantic import BaseModel

import logging
import time
import torch

from src.dqn import create_q_network


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="DQN CartPole API"
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logger = logging.getLogger("dqn-api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

network = create_q_network()

network.load_state_dict(
    torch.load(
        "models/dqn_cartpole.pth",
        map_location="cpu"
    )
)

network.eval()


logger.info(
    "DQN model loaded successfully."
)


# --------------------------------------------------
# Request format
# --------------------------------------------------

class StateInput(BaseModel):

    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "DQN API running"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "DQN-CartPole",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(
    data: StateInput
):

    start_time = time.time()

    state = torch.tensor(
        [[
            data.cart_position,
            data.cart_velocity,
            data.pole_angle,
            data.pole_angular_velocity
        ]],
        dtype=torch.float32
    )


    with torch.no_grad():

        q_values = network(
            state
        )


    action = torch.argmax(
        q_values,
        dim=1
    ).item()


    inference_time = (
        time.time()
        - start_time
    )


    q_left = (
        q_values[0][0]
        .item()
    )

    q_right = (
        q_values[0][1]
        .item()
    )


    logger.info(
        f"Prediction | "
        f"Action={action} | "
        f"Q_left={q_left:.3f} | "
        f"Q_right={q_right:.3f} | "
        f"Inference_time={inference_time:.4f}s"
    )


    return {
        "action": action,
        "q_left": q_left,
        "q_right": q_right,
        "inference_time_seconds": inference_time
    }