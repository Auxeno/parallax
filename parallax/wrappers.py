import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray, PyTree

from typing import Any

from .core import Env, State


class Wrapper:
    """Base wrapper. Subclass this to create custom wrappers."""

    def __init__(self, env: Env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def observation(self, state: State) -> PyTree:
        return self.env.observation(state)

    def reward(self, state: State) -> Float[Array, "..."]:
        return self.env.reward(state)

    def termination(self, state: State) -> Bool[Array, "..."]:
        return self.env.termination(state)

    def truncation(self, state: State) -> Bool[Array, "..."]:
        return self.env.truncation(state)

    def info(self, state: State) -> PyTree:
        return self.env.info(state)

    def reset(self, *, key: PRNGKeyArray) -> State:
        return self.env.reset(key=key).bind(self)

    def step(self, state: State, action: Array) -> State:
        return self.env.step(state, action).bind(self)

    def __getattr__(self, name: str) -> Any:
        env = self.__dict__.get('env')
        if env is not None:
            return getattr(env, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class VmapWrapper(Wrapper):
    """Vectorises an environment over a batch of states using `jax.vmap`."""

    def observation(self, state: State) -> PyTree:
        return jax.vmap(self.env.observation)(state)

    def reward(self, state: State) -> Float[Array, "..."]:
        return jax.vmap(self.env.reward)(state)

    def termination(self, state: State) -> Bool[Array, "..."]:
        return jax.vmap(self.env.termination)(state)

    def truncation(self, state: State) -> Bool[Array, "..."]:
        return jax.vmap(self.env.truncation)(state)

    def info(self, state: State) -> PyTree:
        return jax.vmap(self.env.info)(state)

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: State | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> State:
        if done is None or state is None:
            return jax.vmap(self.env.reset)(key=key).bind(self)
        keys = jax.random.split(key, done.shape[0])
        new_state = jax.vmap(self.env.reset)(key=keys).bind(self)

        def _select(n, o):
            mask = done.reshape(-1, *([1] * (n.ndim - done.ndim)))
            return jnp.where(mask, n, o)

        return jax.tree.map(_select, new_state, state)

    def step(self, state: State, action: Array) -> State:
        return jax.vmap(self.env.step)(state, action).bind(self)

    def __getattr__(self, name: str) -> Any:
        env = self.__dict__.get('env')
        if env is not None:
            attr = getattr(env, name)
            if callable(attr):
                return lambda state: jax.vmap(attr)(state)
            return attr
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class AutoResetWrapper(Wrapper):
    """Automatically resets the environment when an episode ends."""

    def step(self, state: State, action: Array) -> State:
        state = self.env.step(state, action).bind(self)
        done = jnp.any(state.termination | state.truncation)
        key, reset_key = jax.random.split(state.key)
        reset_state = self.env.reset(key=reset_key).bind(self)
        state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, state)
        state.key = key
        return state


class TimeLimit(Wrapper):
    """Truncates episodes that exceed a maximum number of steps."""

    def __init__(self, env: Env, max_steps: int) -> None:
        super().__init__(env)
        self.max_steps = max_steps

    def truncation(self, state: State) -> Bool[Array, "..."]:
        return self.env.truncation(state) | (state.step_count >= self.max_steps)
