from dataclasses import replace

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PRNGKeyArray

from .env import Env
from .struct import EnvState, Timestep


class VmapWrapper:
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: EnvState | None = None,
        timestep: Timestep | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> tuple[EnvState, Timestep]:
        if done is None or state is None or timestep is None:
            return jax.vmap(self.env.reset)(key=key)
        keys = jax.random.split(key, done.shape[0])
        new_state, new_timestep = jax.vmap(self.env.reset)(key=keys)

        def _select(n, o):
            mask = done.reshape(-1, *([1] * (n.ndim - done.ndim)))
            return jnp.where(mask, n, o)

        out_state: EnvState = jax.tree.map(_select, new_state, state)
        out_timestep: Timestep = jax.tree.map(_select, new_timestep, timestep)
        return out_state, out_timestep

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]:
        return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper:
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]:
        return self.env.reset(key=key)

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]:
        state, timestep = self.env.step(state, action)
        done = jnp.any(timestep.termination | timestep.truncation)
        key, reset_key = jax.random.split(state.key)
        reset_state, reset_timestep = self.env.reset(key=reset_key)
        state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, state)
        timestep = jax.tree.map(lambda r, t: jnp.where(done, r, t), reset_timestep, timestep)
        return state, timestep


class TimeLimit:
    def __init__(self, env: Env, max_steps: int):
        self.env = env
        self.max_steps = max_steps
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]:
        return self.env.reset(key=key)

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]:
        state, timestep = self.env.step(state, action)
        truncation = timestep.truncation | (state.step >= self.max_steps)
        timestep = replace(timestep, truncation=truncation)
        return state, timestep
