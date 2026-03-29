"""Adapter for Gymnax environments."""

from typing import Any

import gymnax.environments.environment as gymnax_env
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from parallax.core import State
from parallax.spaces import Box, Discrete, Space


def _convert_space(space: Any) -> Space:
    """Convert a Gymnax space to a Parallax space."""
    if isinstance(space, gymnax_spaces.Discrete):
        return Discrete(n=int(space.n))
    if isinstance(space, gymnax_spaces.Box):
        return Box(
            low=float(jnp.asarray(space.low).min()),
            high=float(jnp.asarray(space.high).max()),
            shape=space.shape,
        )
    raise TypeError(f"Unsupported gymnax space: {type(space)}")


class GymnaxAdapter:
    """Adapt a Gymnax environment to the Parallax protocol.

    Parameters
    ----------
    env : gymnax.environments.environment.Environment
        The Gymnax environment to wrap.

    Examples
    --------
    >>> import gymnax
    >>> env = GymnaxAdapter(gymnax.make("CartPole-v1")[0])
    >>> state = env.reset(key=jax.random.key(0))
    >>> state = env.step(state, action)
    """

    def __init__(self, env: gymnax_env.Environment, params: Any = None) -> None:
        self.env = env
        self.params = params if params is not None else env.default_params
        self.action_space = _convert_space(env.action_space(self.params))
        self.observation_space = _convert_space(env.observation_space(self.params))

    def reset(self, *, key: PRNGKeyArray) -> State:
        obs, env_state = self.env.reset(key, self.params)
        return State(
            env_state=env_state,
            observation=obs,
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={"discount": jnp.float32(1.0)},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state: State, action: Array) -> State:
        key, step_key = jax.random.split(state.key)
        obs, env_state, reward, done, info = self.env.step(
            step_key,
            state.env_state,
            action,
            self.params,
        )
        return State(
            env_state=env_state,
            observation=obs,
            reward=reward,
            termination=done,
            truncation=jnp.bool_(False),
            info=info,
            step_count=state.step_count + 1,
            key=key,
        )
