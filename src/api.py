from fastapi import FastAPI
from pydantic import BaseModel

import torch

from src.dqn import create_q_network


app = FastAPI(
    title="DQN CartPole API"
)


# -----------------------------
# Load trained model
# -----------------------------

network = create_q_network()

network.load_state_dict(
    torch.load(
        "models/dqn_cartpole.pth",
        map_location="cpu"
    )
)

network.eval()


# -----------------------------
# Request format
# -----------------------------

class StateInput(BaseModel):

    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float


# -----------------------------
# Health endpoint
# -----------------------------

@app.get("/")
def health():

    return {
        "status": "DQN API running"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict(
    data: StateInput
):

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

    return {

        "action": action,

        "q_left": q_values[0][0].item(),

        "q_right": q_values[0][1].item()
    }