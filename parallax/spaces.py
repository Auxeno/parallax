from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray, PyTree


class Space(Protocol):
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
        return (self.actions_per_dim.shape[0],)

    def sample(self, *, key: PRNGKeyArray) -> Array:
        keys = jax.random.split(key, self.actions_per_dim.shape[0])
        return jax.vmap(lambda k, n: jax.random.randint(k, (), 0, n))(keys, self.actions_per_dim)


@dataclass
class MultiBinary:
    n: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    def sample(self, *, key: PRNGKeyArray) -> Array:
        return jax.random.bernoulli(key, shape=(self.n,)).astype(jnp.int8)


@dataclass
class PyTreeSpace:
    spaces: PyTree

    def sample(self, *, key: PRNGKeyArray) -> PyTree:
        leaves, treedef = jax.tree.flatten(self.spaces)
        keys = jax.random.split(key, len(leaves))
        samples = [space.sample(key=k) for space, k in zip(leaves, keys)]
        return treedef.unflatten(samples)
