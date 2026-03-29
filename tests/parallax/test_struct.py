"""Tests for the State class."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

from parallax.core import State


def _make_state(**kwargs):
    defaults = dict(
        env_state=jnp.float32(0.0),
        observation=jnp.float32(0.0),
        reward=jnp.float32(0.0),
        termination=jnp.bool_(False),
        truncation=jnp.bool_(False),
        info={},
        step_count=jnp.int32(0),
        key=jax.random.key(0),
    )
    defaults.update(kwargs)
    return State(**defaults)


class TestState:
    def test_is_pytree(self):
        state = _make_state()
        leaves = jtu.tree_leaves(state)
        assert len(leaves) > 0

    def test_flatten_unflatten(self):
        state = _make_state(
            env_state=jnp.float32(42.0), step_count=jnp.int32(5), key=jax.random.key(7)
        )
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

    def test_nested_env_state(self):
        """State.env_state can be a nested pytree."""
        nested = {"pos": jnp.zeros(2), "vel": jnp.ones(3)}
        state = _make_state(env_state=nested)
        leaves = jtu.tree_leaves(state)
        assert len(leaves) >= 4

    def test_jit_compatible(self):
        state = _make_state(env_state=jnp.float32(1.0))

        @jax.jit
        def f(s):
            return State(
                env_state=s.env_state + 1,
                observation=s.observation,
                reward=s.reward,
                termination=s.termination,
                truncation=s.truncation,
                info=s.info,
                step_count=s.step_count + 1,
                key=s.key,
            )

        result = f(state)
        npt.assert_allclose(result.env_state, 2.0)

    def test_vmap_compatible(self):
        state = _make_state(
            env_state=jnp.zeros(4),
            observation=jnp.zeros(4),
            reward=jnp.zeros(4),
            termination=jnp.zeros(4, dtype=jnp.bool_),
            truncation=jnp.zeros(4, dtype=jnp.bool_),
            step_count=jnp.zeros(4, dtype=jnp.int32),
            key=jax.random.split(jax.random.key(0), 4),
        )

        @jax.vmap
        def f(s):
            return State(
                env_state=s.env_state + 1,
                observation=s.observation,
                reward=s.reward,
                termination=s.termination,
                truncation=s.truncation,
                info=s.info,
                step_count=s.step_count,
                key=s.key,
            )

        result = f(state)
        npt.assert_allclose(result.env_state, jnp.ones(4))

    def test_observation(self):
        state = _make_state(observation=jnp.float32(6.0))
        npt.assert_allclose(state.observation, 6.0)

    def test_reward(self):
        state = _make_state(reward=jnp.float32(1.0))
        npt.assert_allclose(state.reward, 1.0)

    def test_termination(self):
        state = _make_state(termination=jnp.bool_(True))
        assert state.termination
        state2 = _make_state(termination=jnp.bool_(False))
        assert not state2.termination

    def test_truncation(self):
        state = _make_state()
        assert not state.truncation

    def test_done(self):
        state = _make_state(termination=jnp.bool_(True))
        assert state.done
        state2 = _make_state(truncation=jnp.bool_(True))
        assert state2.done
        state3 = _make_state()
        assert not state3.done

    def test_info(self):
        state = _make_state()
        assert state.info == {}

    def test_info_with_data(self):
        state = _make_state(info={"score": jnp.float32(15.0)})
        npt.assert_allclose(state.info["score"], 15.0)


class TestSubclass:
    def test_subclass_fields(self):
        """Subclassed State with extra fields works as a pytree."""

        @jax.tree_util.register_dataclass
        @dataclass
        class MaskedState(State):
            action_mask: jax.Array

        state = MaskedState(
            env_state=jnp.float32(0.0),
            observation=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=jax.random.key(0),
            action_mask=jnp.ones(4, dtype=jnp.bool_),
        )
        npt.assert_array_equal(state.action_mask, jnp.ones(4, dtype=jnp.bool_))
        assert not state.done

    def test_subclass_survives_pytree_roundtrip(self):
        """Subclassed State preserves type and fields through flatten/unflatten."""

        @jax.tree_util.register_dataclass
        @dataclass
        class MaskedState(State):
            action_mask: jax.Array

        state = MaskedState(
            env_state=jnp.float32(1.0),
            observation=jnp.float32(1.0),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=jax.random.key(0),
            action_mask=jnp.array([True, False, True, False]),
        )
        leaves, treedef = jtu.tree_flatten(state)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, MaskedState)
        npt.assert_array_equal(reconstructed.action_mask, state.action_mask)
