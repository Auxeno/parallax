from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray, PyTree


class Space(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def sample(self, *, key: PRNGKeyArray) -> Array | PyTree: ...


@dataclass
class Discrete:
    n: int
    shape: tuple[int, ...] = ()

    def sample(self, *, key: PRNGKeyArray) -> Array:
        return jax.random.randint(key, self.shape, 0, self.n)


@dataclass
class Box:
    low: float | Array
    high: float | Array
    shape: tuple[int, ...]

    def sample(self, *, key: PRNGKeyArray) -> Array:
        return jax.random.uniform(key, self.shape, minval=self.low, maxval=self.high)


@dataclass
class MultiDiscrete:
    actions_per_dim: Int[Array, "..."]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.actions_per_dim.shape

    def sample(self, *, key: PRNGKeyArray) -> Array:
        return jax.random.randint(key, self.actions_per_dim.shape, 0, self.actions_per_dim)


@dataclass
class MultiBinary:
    n: int | tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,) if isinstance(self.n, int) else self.n

    def sample(self, *, key: PRNGKeyArray) -> Array:
        return jax.random.bernoulli(key, shape=self.shape).astype(jnp.int8)


@dataclass
class PyTreeSpace:
    spaces: PyTree

    @property
    def shape(self) -> PyTree:
        return jax.tree.map(lambda s: s.shape, self.spaces)

    def sample(self, *, key: PRNGKeyArray) -> PyTree:
        leaves, treedef = jax.tree.flatten(self.spaces)
        keys = jax.random.split(key, len(leaves))
        samples = [space.sample(key=k) for space, k in zip(leaves, keys)]
        return treedef.unflatten(samples)


def stack_space(space: Space, num: int) -> Space:
    """Return a copy of `space` with a leading dim of size `num`."""
    if isinstance(space, Discrete):
        return Discrete(space.n, (num, *space.shape))
    if isinstance(space, Box):
        return Box(space.low, space.high, (num, *space.shape))
    if isinstance(space, MultiDiscrete):
        apd = space.actions_per_dim
        return MultiDiscrete(jnp.broadcast_to(apd, (num, *apd.shape)))
    if isinstance(space, MultiBinary):
        return MultiBinary((num, *space.shape))
    if isinstance(space, PyTreeSpace):
        return PyTreeSpace(jax.tree.map(lambda s: stack_space(s, num), space.spaces))
    raise TypeError(f"Cannot stack space of type {type(space).__name__}")
