from dataclasses import dataclass
from typing import Protocol

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
