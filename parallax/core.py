from typing import Any, Protocol, Self

import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .spaces import Space


class Env(Protocol):
    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> "State": ...
    def step(self, state: "State", action: Array) -> "State": ...

    def observation(self, state: "State") -> PyTree: ...
    def reward(self, state: "State") -> Float[Array, "..."]: ...
    def termination(self, state: "State") -> Bool[Array, "..."]: ...
    def truncation(self, state: "State") -> Bool[Array, "..."]: ...
    def info(self, state: "State") -> PyTree: ...


@jax.tree_util.register_pytree_node_class
class State:
    def __init__(
        self,
        env: Env,
        *,
        env_state: PyTree,
        step_count: Int[Array, ""],
        key: PRNGKeyArray,
    ) -> None:
        self._env = env
        self.env_state = env_state
        self.step_count = step_count
        self.key = key

    @property
    def observation(self) -> PyTree:
        return self._env.observation(self)

    @property
    def reward(self) -> Float[Array, "..."]:
        return self._env.reward(self)

    @property
    def termination(self) -> Bool[Array, "..."]:
        return self._env.termination(self)

    @property
    def truncation(self) -> Bool[Array, "..."]:
        return self._env.truncation(self)

    @property
    def info(self) -> PyTree:
        return self._env.info(self)

    @property
    def done(self) -> Bool[Array, "..."]:
        return self.termination | self.truncation

    def __getattr__(self, name: str) -> Any:
        """Forwards attribute access to the env, so custom env methods work as state properties."""
        env = self.__dict__.get("_env")
        if env is not None:
            method = getattr(env, name, None)
            if callable(method):
                return method(self)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def bind(self, env: Env) -> Self:
        """Rebinds this state so that wrappers can override how properties are computed."""
        return type(self)(env, env_state=self.env_state, step_count=self.step_count, key=self.key)

    def tree_flatten(self) -> tuple[tuple[PyTree, Int[Array, ""], PRNGKeyArray], Env]:
        children = (self.env_state, self.step_count, self.key)
        return children, self._env

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Env,
        children: tuple[PyTree, Int[Array, ""], PRNGKeyArray],
    ) -> "State":
        env_state, step_count, key = children
        return cls(aux_data, env_state=env_state, step_count=step_count, key=key)
