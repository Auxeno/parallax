"""Adapter for Brax environments."""

from typing import Any

import brax.envs.wrappers.training as brax_training
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from parallax.core import State
from parallax.spaces import Box, Space


class BraxAdapter:
    """Adapt a Brax environment to the Parallax protocol."""

    def __init__(self, env: Any) -> None:
        # Strip Brax wrappers, extracting episode_length before discarding
        episode_length = 1000
        if isinstance(env, brax_training.AutoResetWrapper):
            env = env.env
        if isinstance(env, brax_training.EpisodeWrapper):
            episode_length = env.episode_length
            env = env.env

        self.env = env
        self.action_space: Space = Box(-1.0, 1.0, (env.action_size,))
        self.observation_space: Space = Box(-jnp.inf, jnp.inf, (env.observation_size,))
        self.episode_length = episode_length

    def reset(self, *, key: PRNGKeyArray) -> State:
        brax_state = self.env.reset(key)
        return State(
            env_state=brax_state,
            observation=brax_state.obs,
            reward=jnp.float32(brax_state.reward),
            termination=jnp.bool_(brax_state.done),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state: State, action: Array) -> State:
        brax_state = self.env.step(state.env_state, action)
        step_count = state.step_count + 1
        return State(
            env_state=brax_state,
            observation=brax_state.obs,
            reward=jnp.float32(brax_state.reward),
            termination=jnp.bool_(brax_state.done),
            truncation=jnp.bool_(step_count >= self.episode_length),
            info={},
            step_count=step_count,
            key=state.key,
        )
