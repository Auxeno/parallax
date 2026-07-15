"""Environment protocols and state structs.

- `State` / `MARLState`: pytree states returned by `reset` and `step`
- `Agents`: per-agent view stored on `MARLState`
- `StateT` / `MARLStateT`: TypeVars for state-generic code, as in `Wrapper[StateT]`
- `Env` / `VectorEnv`: single-agent protocols, plain and vectorised
- `MARLEnv` / `MARLVectorEnv`: their multi-agent counterparts

The protocols are structural, an env satisfies them without subclassing.
"""

from dataclasses import dataclass
from typing import Protocol

import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from typing_extensions import TypeVar

from .spaces import Space


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


StateT = TypeVar("StateT", bound=State | MARLState, default=State)
MARLStateT = TypeVar("MARLStateT", bound=MARLState, default=MARLState)


class Env(Protocol[StateT]):
    """Parallax environment protocol. Bare `Env` means `Env[State]`."""

    @property
    def action_space(self) -> Space: ...

    @property
    def observation_space(self) -> Space: ...

    def reset(self, *, key: PRNGKeyArray) -> StateT: ...
    def step(self, state: StateT, action: PyTree) -> StateT: ...


class VectorEnv(Env[StateT], Protocol[StateT]):
    """Vectorised environment protocol with selective reset."""

    num_envs: int
    """Static number of parallel environments."""

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: StateT | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> StateT: ...


class MARLEnv(Env[MARLStateT], Protocol[MARLStateT]):
    """Parallax multi-agent environment protocol. Bare `MARLEnv` means `MARLEnv[MARLState]`."""

    num_agents: int
    """Static number of agent slots."""


class MARLVectorEnv(MARLEnv[MARLStateT], VectorEnv[MARLStateT], Protocol[MARLStateT]):
    """Vectorised multi-agent environment protocol with selective reset."""

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: MARLStateT | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> MARLStateT: ...
