"""Adapter for MuJoCo Playground (MJX) environments."""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from parallax.core import State
from parallax.spaces import Box, PyTreeSpace, Space


class MJXAdapter:
    """Adapt a MuJoCo Playground environment to the Parallax protocol.

    Parameters
    ----------
    env : MjxEnv
        A MuJoCo Playground environment, e.g. from `registry.load()`.

    Examples
    --------
    >>> from mujoco_playground import registry
    >>> env = MJXAdapter(registry.load("CartpoleBalance", config_overrides={"impl": "jax"}))
    >>> state = env.reset(key=jax.random.key(0))
    >>> state = env.step(state, action)
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.episode_length = env._config.episode_length
        obs_size = env.observation_size
        if isinstance(obs_size, int):
            self.observation_space: Space = Box(-jnp.inf, jnp.inf, (obs_size,))
        elif isinstance(obs_size, dict):
            self.observation_space: Space = PyTreeSpace(
                {k: Box(-jnp.inf, jnp.inf, v) for k, v in obs_size.items()}
            )
        else:
            raise TypeError(f"Unsupported observation_size type: {type(obs_size)}")
        self.action_space: Space = Box(-1.0, 1.0, (env.action_size,))

    def reset(self, *, key: PRNGKeyArray) -> State:
        mjx_state = self.env.reset(key)
        return State(
            env_state=mjx_state,
            observation=mjx_state.obs,
            reward=jnp.float32(mjx_state.reward),
            termination=jnp.bool_(mjx_state.done),
            truncation=jnp.bool_(False),
            info=mjx_state.metrics,
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state: State, action: Array) -> State:
        mjx_state = self.env.step(state.env_state, action)
        step_count = state.step_count + 1
        return State(
            env_state=mjx_state,
            observation=mjx_state.obs,
            reward=jnp.float32(mjx_state.reward),
            termination=jnp.bool_(mjx_state.done),
            truncation=jnp.bool_(step_count >= self.episode_length),
            info=mjx_state.metrics,
            step_count=step_count,
            key=state.key,
        )
