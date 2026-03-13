from typing import Protocol

from jaxtyping import Array, PRNGKeyArray

from .spaces import Space
from .struct import EnvState, Timestep


class Env(Protocol):
    action_space: Space
    observation_space: Space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]: ...
    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]: ...
