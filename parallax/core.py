from dataclasses import dataclass
from typing import Protocol

import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .spaces import Space


class Env(Protocol):
    """Parallax environment protocol."""

    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> "State": ...
    def step(self, state: "State", action: Array) -> "State": ...


class VectorEnv(Env, Protocol):
    """Vectorised environment protocol with selective reset."""

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
    """Environment state returned by `reset` and `step`."""

    env_state: PyTree
    """Raw environment data, any pytree."""

    observation: PyTree
    """What the agent sees."""

    reward: Float[Array, "..."]
    """Scalar reward."""

    termination: Bool[Array, "..."]
    """The episode ended naturally."""

    truncation: Bool[Array, "..."]
    """The episode was cut short."""

    info: PyTree
    """Extra environment metadata."""

    step_count: Int[Array, ""]
    """Steps taken this episode."""

    key: PRNGKeyArray
    """RNG key."""

    @property
    def done(self) -> Bool[Array, "..."]:
        return self.termination | self.truncation


class MARLEnv(Protocol):
    """Parallax multi-agent environment protocol."""

    num_agents: int
    """Static number of agent slots."""

    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> "MARLState": ...
    def step(self, state: "MARLState", action: PyTree) -> "MARLState": ...


class MARLVectorEnv(MARLEnv, Protocol):
    """Vectorised multi-agent environment protocol with selective reset."""

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: "MARLState | None" = None,
        done: "Bool[Array, '...'] | None" = None,
    ) -> "MARLState": ...


@jax.tree_util.register_dataclass
@dataclass
class Agents:
    """Per-agent view of a multi-agent state, leaves lead with `num_agents`."""

    observation: PyTree
    """What each agent sees, zeros for inactive agents."""

    active: Bool[Array, "... A"]
    """Which agents act on the next step."""

    reward: Float[Array, "... A"]
    """Reward for each agent, zeros in inactive slots."""

    action_mask: PyTree | None = None
    """Optional legal action mask for each agent."""


@jax.tree_util.register_dataclass
@dataclass
class MARLState:
    """Multi-agent environment state returned by `reset` and `step`."""

    env_state: PyTree
    """Raw environment data, any pytree."""

    agents: Agents
    """Per-agent observations, rewards and activity."""

    termination: Bool[Array, "..."]
    """The MDP ended naturally."""

    truncation: Bool[Array, "..."]
    """The MDP was cut short."""

    info: PyTree
    """Extra environment metadata."""

    step_count: Int[Array, ""]
    """Steps taken this episode."""

    key: PRNGKeyArray
    """RNG key."""

    global_observation: PyTree | None = None
    """Optional full-state view for centralised critics."""

    @property
    def done(self) -> Bool[Array, "..."]:
        return self.termination | self.truncation
