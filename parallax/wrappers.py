from dataclasses import replace
from typing import Any

import jax
from jaxtyping import Array, Bool, PRNGKeyArray

from .core import Env, State


class Wrapper:
    """Base wrapper. Subclass this to create custom wrappers."""

    def __init__(self, env: Env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, key: PRNGKeyArray) -> State:
        return self.env.reset(key=key)

    def step(self, state: State, action: Array) -> State:
        return self.env.step(state, action)

    def __getattr__(self, name: str) -> Any:
        env = self.__dict__.get("env")
        if env is not None:
            return getattr(env, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class VmapWrapper(Wrapper):
    """Vectorise an environment over a batch of states using `jax.vmap`.

    Parameters
    ----------
    env : Env
        The environment to vectorise.
    num_envs : int
        Number of parallel environments.

    Examples
    --------
    >>> env = VmapWrapper(MyEnv(), num_envs=128)
    >>> state = env.reset(key=jax.random.key(0))
    >>> state = env.step(state, actions)

    Selectively reset only finished environments:

    >>> state = env.reset(key=jax.random.key(1), state=state, done=state.done)
    """

    def __init__(self, env: Env, num_envs: int) -> None:
        super().__init__(env)
        self.num_envs = num_envs

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: State | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> State:
        """Reset all environments, or only those where `done=True`.

        When `state` and `done` are omitted, all environments are reset.
        When both are provided, only environments where `done=True` are reset.

        Parameters
        ----------
        key : PRNGKeyArray
            A single RNG key, split internally across environments.
        state : State, optional
            Current batched state. Required for selective resets.
        done : Bool[Array, "num_envs"], optional
            Boolean mask indicating which environments to reset.

        Returns
        -------
        State
            Batched state with leading dim `num_envs` on all leaves.
        """
        keys = jax.random.split(key, self.num_envs)
        reset_state = jax.vmap(self.env.reset)(key=keys)

        if done is None or state is None:
            return reset_state

        return jax.vmap(lambda d, r, s: jax.lax.cond(d, lambda: r, lambda: s))(
            done,
            reset_state,
            state,
        )

    def step(self, state: State, action: Array) -> State:
        """Step all environments in parallel.

        Parameters
        ----------
        state : State
            Batched state with leading dim `num_envs`.
        action : Array
            Batched actions with leading dim `num_envs`.

        Returns
        -------
        State
            Updated batched state.
        """
        return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper(Wrapper):
    """Automatically resets the environment when an episode ends."""

    def step(self, state: State, action: Array) -> State:
        state = self.env.step(state, action)
        key, reset_key = jax.random.split(state.key)
        reset_state = self.env.reset(key=reset_key)
        state = jax.lax.cond(state.done, lambda: reset_state, lambda: state)
        return replace(state, key=key)


class TimeLimit(Wrapper):
    """Truncates episodes that exceed a maximum number of steps."""

    def __init__(self, env: Env, max_steps: int) -> None:
        super().__init__(env)
        self.max_steps = max_steps

    def step(self, state: State, action: Array) -> State:
        state = self.env.step(state, action)
        return replace(state, truncation=state.truncation | (state.step_count >= self.max_steps))
