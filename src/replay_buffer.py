import random


def create_replay_buffer(capacity):
    return []


def add_experience(buffer, state, action, reward, next_state, done, capacity):

    experience = (
        state,
        action,
        reward,
        next_state,
        done
    )

    buffer.append(experience)

    if len(buffer) > capacity:
        buffer.pop(0)


def sample_experiences(buffer, batch_size):
    return random.sample(buffer, batch_size)

