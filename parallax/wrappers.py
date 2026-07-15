"""Composable environment wrappers.

- `Wrapper`: base class for custom wrappers, forwards to the wrapped env
- `VmapWrapper`: batch an env over `num_envs` with `jax.vmap`, selective reset
- `AutoResetWrapper`: restart episodes as they end
- `TimeLimit`: truncate episodes after `max_steps`

Wrappers preserve the wrapped env's state type, so a wrapped MARL env
still satisfies the MARL protocols.
"""

from dataclasses import replace
from typing import Any, Generic

import jax
from jaxtyping import Array, Bool, PRNGKeyArray, PyTree

from .core import Env, StateT


class Wrapper(Generic[StateT]):
    """Base wrapper. Subclass this to create custom wrappers."""

    num_agents: int

    def __init__(self, env: Env[StateT]) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_agents = getattr(env, "num_agents", 1)

    def reset(self, *, key: PRNGKeyArray, **kwargs: Any) -> StateT:
        return self.env.reset(key=key, **kwargs)

    def step(self, state: StateT, action: PyTree) -> StateT:
        return self.env.step(state, action)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped env."""
        env = self.__dict__.get("env")
        if env is not None:
            return getattr(env, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class VmapWrapper(Wrapper[StateT]):
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

    def __init__(self, env: Env[StateT], num_envs: int) -> None:
        super().__init__(env)
        self.num_envs = num_envs

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: StateT | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> StateT:
        """Reset all environments, or only those where `done=True`.

        When `state` and `done` are omitted, all environments are reset.
        When both are provided, only environments where `done=True` are reset.
        Providing one without the other raises a `ValueError`.

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
        if (state is None) != (done is None):
            raise ValueError("Selective reset requires both `state` and `done`")

        keys = jax.random.split(key, self.num_envs)
        reset_state = jax.vmap(self.env.reset)(key=keys)

        if done is None or state is None:
            return reset_state

        return jax.vmap(lambda d, r, s: jax.lax.cond(d, lambda: r, lambda: s))(
            done,
            reset_state,
            state,
        )

    def step(self, state: StateT, action: PyTree) -> StateT:
        """Step all environments in parallel.

        Parameters
        ----------
        state : State
            Batched state with leading dim `num_envs`.
        action : PyTree
            Batched actions with leading dim `num_envs` on all leaves.

        Returns
        -------
        State
            Updated batched state.
        """
        return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper(Wrapper[StateT]):
    """Automatically resets the environment when an episode ends."""

    def step(self, state: StateT, action: PyTree) -> StateT:
        state = self.env.step(state, action)
        key, reset_key = jax.random.split(state.key)
        reset_state = self.env.reset(key=reset_key)
        state = jax.lax.cond(state.done, lambda: reset_state, lambda: state)
        return replace(state, key=key)


class TimeLimit(Wrapper[StateT]):
    """Truncates episodes that exceed a maximum number of steps."""

    def __init__(self, env: Env[StateT], max_steps: int) -> None:
        super().__init__(env)
        self.max_steps = max_steps

    def step(self, state: StateT, action: PyTree) -> StateT:
        state = self.env.step(state, action)
        return replace(state, truncation=state.truncation | (state.step_count >= self.max_steps))
