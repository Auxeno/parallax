"""Tests for the State class."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

from parallax.core import State
from parallax.spaces import Box, Discrete


class _DummyEnv:
    """Minimal env for testing State construction and lazy properties."""

    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=10.0, shape=())

    def reset(self, *, key):
        return State(self, env_state=jnp.float32(0.0), step_count=jnp.int32(0), key=key)

    def step(self, state, action):
        return State(self, env_state=state.env_state + 1.0, step_count=state.step_count + 1, key=state.key)

    def observation(self, state):
        return state.env_state * 2.0

    def reward(self, state):
        return jnp.float32(1.0)

    def termination(self, state):
        return state.env_state >= 5.0

    def truncation(self, state):
        return jnp.bool_(False)

    def info(self, state):
        return {}


_ENV = _DummyEnv()


def _make_state(**kwargs):
    defaults = dict(env_state=jnp.float32(0.0), step_count=jnp.int32(0), key=jax.random.key(0))
    defaults.update(kwargs)
    return State(_ENV, **defaults)


class TestState:
    def test_is_pytree(self):
        state = _make_state()
        leaves = jtu.tree_leaves(state)
        assert len(leaves) > 0

    def test_flatten_unflatten(self):
        state = _make_state(env_state=jnp.float32(42.0), step_count=jnp.int32(5), key=jax.random.key(7))
        leaves, treedef = jtu.tree_flatten(state)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, State)
        npt.assert_array_equal(reconstructed.env_state, state.env_state)
        npt.assert_array_equal(reconstructed.step_count, state.step_count)

    def test_tree_map(self):
        state = _make_state(env_state=jnp.float32(1.0), step_count=jnp.int32(2))
        mapped = jax.tree.map(lambda x: x, state)
        assert isinstance(mapped, State)
        npt.assert_allclose(mapped.env_state, 1.0)
        npt.assert_array_equal(mapped.step_count, 2)

    def test_nested_state(self):
        """State.env_state can be a nested pytree."""
        nested = {"pos": jnp.zeros(2), "vel": jnp.ones(3)}
        state = _make_state(env_state=nested)
        leaves = jtu.tree_leaves(state)
        # pos(2) + vel(3) + step_count + key = at least 4 leaves
        assert len(leaves) >= 4

    def test_jit_compatible(self):
        state = _make_state(env_state=jnp.float32(1.0))
        result = jax.jit(lambda s: State(_ENV, env_state=s.env_state + 1, step_count=s.step_count + 1, key=s.key))(
            state
        )
        npt.assert_allclose(result.env_state, 2.0)

    def test_vmap_compatible(self):
        states = State(
            _ENV,
            env_state=jnp.zeros(4),
            step_count=jnp.zeros(4, dtype=jnp.int32),
            key=jax.random.split(jax.random.key(0), 4),
        )
        result = jax.vmap(lambda s: State(_ENV, env_state=s.env_state + 1, step_count=s.step_count, key=s.key))(
            states
        )
        npt.assert_allclose(result.env_state, jnp.ones(4))

    def test_key_is_pytree_leaf(self):
        state = _make_state(key=jax.random.key(42))
        leaves = jtu.tree_leaves(state)
        # env_state, step_count, key = 3 leaves
        assert len(leaves) == 3

    def test_env_in_aux_not_leaves(self):
        """The _env reference should be in aux data, not in leaves."""
        state = _make_state()
        leaves = jtu.tree_leaves(state)
        assert _ENV not in leaves

    def test_lazy_observation(self):
        state = _make_state(env_state=jnp.float32(3.0))
        npt.assert_allclose(state.observation, 6.0)

    def test_lazy_reward(self):
        state = _make_state()
        npt.assert_allclose(state.reward, 1.0)

    def test_lazy_termination(self):
        state = _make_state(env_state=jnp.float32(5.0))
        assert state.termination
        state2 = _make_state(env_state=jnp.float32(1.0))
        assert not state2.termination

    def test_lazy_truncation(self):
        state = _make_state()
        assert not state.truncation

    def test_done(self):
        state = _make_state(env_state=jnp.float32(5.0))
        assert state.done
        state2 = _make_state(env_state=jnp.float32(1.0))
        assert not state2.done

    def test_bind(self):
        state = _make_state(env_state=jnp.float32(3.0))

        class _OtherEnv(_DummyEnv):
            def observation(self, state):
                return state.env_state * 10.0

        other = _OtherEnv()
        swapped = state.bind(other)
        npt.assert_allclose(swapped.observation, 30.0)
        npt.assert_allclose(swapped.env_state, state.env_state)

    def test_custom_property_via_env_method(self):
        """Custom env methods are accessible as state properties via __getattr__."""

        class _CustomEnv(_DummyEnv):
            def action_mask(self, state):
                return jnp.ones(3, dtype=jnp.bool_)

        env = _CustomEnv()
        state = State(env, env_state=jnp.float32(0.0), step_count=jnp.int32(0), key=jax.random.key(0))
        npt.assert_array_equal(state.action_mask, jnp.ones(3, dtype=jnp.bool_))
        # Core properties still work
        npt.assert_allclose(state.observation, 0.0)

    def test_custom_property_survives_pytree_roundtrip(self):
        """Custom env properties work after flatten/unflatten."""

        class _CustomEnv(_DummyEnv):
            def action_mask(self, state):
                return state.env_state * jnp.ones(3)

        env = _CustomEnv()
        state = State(env, env_state=jnp.float32(2.0), step_count=jnp.int32(0), key=jax.random.key(0))
        leaves, treedef = jtu.tree_flatten(state)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        npt.assert_allclose(reconstructed.action_mask, jnp.full(3, 2.0))

    def test_custom_property_missing_raises(self):
        """Accessing a nonexistent property raises AttributeError."""
        state = _make_state()
        import pytest
        with pytest.raises(AttributeError):
            state.nonexistent_property

    def test_lazy_info(self):
        state = _make_state()
        assert state.info == {}

    def test_info_delegates_to_env(self):
        class _InfoEnv(_DummyEnv):
            def info(self, state):
                return {"score": state.env_state * 3.0}

        env = _InfoEnv()
        state = State(env, env_state=jnp.float32(5.0), step_count=jnp.int32(0), key=jax.random.key(0))
        npt.assert_allclose(state.info["score"], 15.0)
