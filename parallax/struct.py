from dataclasses import dataclass

import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree


@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    state: PyTree
    """Base environment/simulator/game state."""

    step: Int[Array, ""]
    """Current timestep count."""

    key: PRNGKeyArray
    """RNG key used in environment."""


@jax.tree_util.register_dataclass
@dataclass
class Timestep:
    """Basic Gymnasium/PettingZoo style struct containing timestep data."""

    observation: Float[Array, "..."] | PyTree
    """Observation from environment state."""

    reward: Float[Array, "..."]
    """Reward received on entering this state."""

    termination: Bool[Array, "..."]
    """This state is terminal."""

    truncation: Bool[Array, "..."]
    """This state is truncated."""

    info: PyTree
    """Container for other information, could be logs from environment."""
