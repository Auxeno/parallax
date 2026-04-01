from dataclasses import dataclass
from typing import Protocol

import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .spaces import Space


class Env(Protocol):
    """Basic Parallax environment protocol."""

    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> "State": ...
    def step(self, state: "State", action: Array) -> "State": ...


class VectorEnv(Env, Protocol):
    """Vectorised Parallax environment protocol."""

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: "State | None" = None,
        done: "Bool[Array, '...'] | None" = None,
    ) -> "State": ...


@jax.tree_util.register_dataclass
@dataclass
class State:
    env_state: PyTree
    observation: PyTree
    reward: Float[Array, "..."]
    termination: Bool[Array, "..."]
    truncation: Bool[Array, "..."]
    info: PyTree
    step_count: Int[Array, ""]
    key: PRNGKeyArray

    @property
    def done(self) -> Bool[Array, "..."]:
        return self.termination | self.truncation
