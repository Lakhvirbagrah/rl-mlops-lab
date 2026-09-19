import torch

from src.dqn import (
    create_q_network,
    decay_epsilon
)

from src.replay_buffer import (
    create_replay_buffer,
    add_experience,
    sample_experiences
)


def test_q_network_output_shape():

    network = create_q_network()

    state = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32
    )

    output = network(state)

    assert output.shape == (1, 2)


def test_epsilon_decay():

    epsilon = 1.0

    new_epsilon = decay_epsilon(
        epsilon,
        decay_rate=0.99,
        min_epsilon=0.05
    )

    assert new_epsilon == 0.99


def test_epsilon_minimum():

    epsilon = 0.051

    new_epsilon = decay_epsilon(
        epsilon,
        decay_rate=0.5,
        min_epsilon=0.05
    )

    assert new_epsilon == 0.05


def test_replay_buffer_sampling():

    replay_buffer = create_replay_buffer(
        100
    )

    state = [0.0, 0.0, 0.0, 0.0]
    action = 1
    reward = 1.0
    next_state = [0.1, 0.0, 0.0, 0.0]
    done = False

    for _ in range(10):

        add_experience(
            replay_buffer,
            state,
            action,
            reward,
            next_state,
            done,
            100
        )

    batch = sample_experiences(
        replay_buffer,
        5
    )

    assert len(batch) == 5