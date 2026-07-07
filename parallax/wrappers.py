from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar

import jax
from jaxtyping import Array, Bool, PRNGKeyArray

from .core import MARLState, State
from .spaces import Space

# Same state type in, same state type out, bound to the wrapped env at construction
StateT = TypeVar("StateT", bound=State | MARLState)


class EnvLike(Protocol[StateT]):
    """Structural env protocol binding a wrapper to its environment's state type."""

    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> StateT: ...
    def step(self, state: StateT, action: Any) -> StateT: ...


class Wrapper(Generic[StateT]):
    """Base wrapper. Subclass this to create custom wrappers."""

    def __init__(self, env: EnvLike[StateT]) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, key: PRNGKeyArray) -> StateT:
        return self.env.reset(key=key)

    def step(self, state: StateT, action: Array) -> StateT:
        return self.env.step(state, action)

    def __getattr__(self, name: str) -> Any:
        env = self.__dict__.get("env")
        if env is not None:
            return getattr(env, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class VmapWrapper(Wrapper[StateT]):
    """Vectorise an environment over a batch of states using `jax.vmap`.

    Parameters
    ----------
    env : EnvLike
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

    def __init__(self, env: EnvLike[StateT], num_envs: int) -> None:
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

    def step(self, state: StateT, action: Array) -> StateT:
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


class AutoResetWrapper(Wrapper[StateT]):
    """Automatically resets the environment when an episode ends."""

    def step(self, state: StateT, action: Array) -> StateT:
        state = self.env.step(state, action)
        key, reset_key = jax.random.split(state.key)
        reset_state = self.env.reset(key=reset_key)
        state = jax.lax.cond(state.done, lambda: reset_state, lambda: state)
        return replace(state, key=key)


class TimeLimit(Wrapper[StateT]):
    """Truncates episodes that exceed a maximum number of steps."""

    def __init__(self, env: EnvLike[StateT], max_steps: int) -> None:
        super().__init__(env)
        self.max_steps = max_steps

    def step(self, state: StateT, action: Array) -> StateT:
        state = self.env.step(state, action)
        return replace(state, truncation=state.truncation | (state.step_count >= self.max_steps))
