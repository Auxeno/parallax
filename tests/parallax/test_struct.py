"""Tests for EnvState and Timestep dataclasses."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

from parallax.struct import EnvState, Timestep


class TestEnvState:
    def test_is_pytree(self):
        state = EnvState(state=jnp.float32(0.0), step=jnp.int32(0), key=jax.random.key(0))
        leaves = jtu.tree_leaves(state)
        assert len(leaves) > 0

    def test_flatten_unflatten(self):
        state = EnvState(state=jnp.float32(42.0), step=jnp.int32(5), key=jax.random.key(7))
        leaves, treedef = jtu.tree_flatten(state)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, EnvState)
        npt.assert_array_equal(reconstructed.state, state.state)
        npt.assert_array_equal(reconstructed.step, state.step)

    def test_tree_map(self):
        state = EnvState(state=jnp.float32(1.0), step=jnp.int32(2), key=jax.random.key(0))
        mapped = jax.tree.map(lambda x: x, state)
        assert isinstance(mapped, EnvState)
        npt.assert_allclose(mapped.state, 1.0)
        npt.assert_array_equal(mapped.step, 2)

    def test_nested_state(self):
        """EnvState.state can be a nested pytree."""
        nested = {"pos": jnp.zeros(2), "vel": jnp.ones(3)}
        state = EnvState(state=nested, step=jnp.int32(0), key=jax.random.key(0))
        leaves = jtu.tree_leaves(state)
        # pos(2 elems) + vel(3 elems) + step + key = at least 4 leaves
        assert len(leaves) >= 4

    def test_jit_compatible(self):
        state = EnvState(state=jnp.float32(1.0), step=jnp.int32(0), key=jax.random.key(0))
        result = jax.jit(lambda s: EnvState(state=s.state + 1, step=s.step + 1, key=s.key))(state)
        npt.assert_allclose(result.state, 2.0)

    def test_vmap_compatible(self):
        states = EnvState(
            state=jnp.zeros(4),
            step=jnp.zeros(4, dtype=jnp.int32),
            key=jax.random.split(jax.random.key(0), 4),
        )
        result = jax.vmap(lambda s: EnvState(state=s.state + 1, step=s.step, key=s.key))(states)
        npt.assert_allclose(result.state, jnp.ones(4))

    def test_key_is_pytree_leaf(self):
        """EnvState.key is included in pytree leaves."""
        state = EnvState(state=jnp.float32(0.0), step=jnp.int32(0), key=jax.random.key(42))
        leaves = jtu.tree_leaves(state)
        assert len(leaves) == 3


class TestTimestep:
    def test_is_pytree(self):
        ts = Timestep(
            observation=jnp.zeros(4),
            reward=jnp.float32(1.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        leaves = jtu.tree_leaves(ts)
        assert len(leaves) > 0

    def test_flatten_unflatten(self):
        ts = Timestep(
            observation=jnp.array([1.0, 2.0]),
            reward=jnp.float32(3.0),
            termination=jnp.bool_(True),
            truncation=jnp.bool_(False),
            info=jnp.float32(99.0),
        )
        leaves, treedef = jtu.tree_flatten(ts)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, Timestep)
        npt.assert_array_equal(reconstructed.observation, ts.observation)
        npt.assert_array_equal(reconstructed.reward, ts.reward)
        npt.assert_array_equal(reconstructed.termination, ts.termination)

    def test_tree_map(self):
        ts = Timestep(
            observation=jnp.ones(2),
            reward=jnp.float32(1.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=jnp.float32(0.0),
        )
        doubled = jax.tree.map(lambda x: x * 2, ts)
        npt.assert_allclose(doubled.reward, 2.0)
        npt.assert_allclose(doubled.observation, jnp.full(2, 2.0))

    def test_jit_compatible(self):
        ts = Timestep(
            observation=jnp.zeros(3),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        result = jax.jit(
            lambda t: Timestep(
                observation=t.observation + 1,
                reward=t.reward + 5.0,
                termination=t.termination,
                truncation=t.truncation,
                info=t.info,
            )
        )(ts)
        npt.assert_allclose(result.reward, 5.0)
        npt.assert_allclose(result.observation, jnp.ones(3))

    def test_pytree_info(self):
        """info field can be a nested pytree."""
        info = {"logs": jnp.zeros(2), "metrics": {"score": jnp.float32(0.0)}}
        ts = Timestep(
            observation=jnp.zeros(1),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=info,
        )
        leaves = jtu.tree_leaves(ts)
        # obs(1) + reward + term + trunc + logs(2 elems) + score = 7 leaves
        assert len(leaves) >= 6

    def test_vmap_compatible(self):
        batch = Timestep(
            observation=jnp.zeros((4, 3)),
            reward=jnp.zeros(4),
            termination=jnp.zeros(4, dtype=jnp.bool_),
            truncation=jnp.zeros(4, dtype=jnp.bool_),
            info=jnp.zeros(4),
        )
        result = jax.vmap(
            lambda t: Timestep(
                observation=t.observation + 1,
                reward=t.reward + 1.0,
                termination=t.termination,
                truncation=t.truncation,
                info=t.info,
            )
        )(batch)
        npt.assert_allclose(result.reward, jnp.ones(4))
