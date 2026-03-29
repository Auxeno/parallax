"""Cross-environment generic tests.

Every test here runs once per env registered in conftest (CountingEnv, GridWorldEnv).
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from conftest import assert_trees_close, assert_trees_equal
from parallax.core import State


def test_reset_returns_state(env, key):
    """reset returns a State."""
    state = env.reset(key=key)
    assert isinstance(state, State)


def test_step_returns_state(env, key):
    """step returns a State."""
    state = env.reset(key=key)
    action = env.action_space.sample(key=key)
    new_state = env.step(state, action)
    assert isinstance(new_state, State)


def test_step_increments(env, key):
    """step count increments each step."""
    state = env.reset(key=key)
    assert state.step_count == 0
    action = env.action_space.sample(key=key)
    state = env.step(state, action)
    assert state.step_count == 1


def test_reset_not_done(env, key):
    """Initial state is neither terminal nor truncated."""
    state = env.reset(key=key)
    assert not state.termination
    assert not state.truncation


def test_reset_determinism(env, key):
    """Same key produces identical reset outputs."""
    s1 = env.reset(key=key)
    s2 = env.reset(key=key)
    assert_trees_equal(s1, s2)


def test_step_determinism(env, key):
    """Same state and action produce identical step outputs."""
    state = env.reset(key=key)
    action = env.action_space.sample(key=key)
    s1 = env.step(state, action)
    s2 = env.step(state, action)
    assert_trees_equal(s1, s2)


def test_jit_reset(env, key):
    """jax.jit produces the same reset output as eager execution."""
    expected = env.reset(key=key)
    result = jax.jit(env.reset)(key=key)
    assert_trees_close(expected, result)


def test_jit_step(env, key):
    """jax.jit produces the same step output as eager execution."""
    state = env.reset(key=key)
    action = env.action_space.sample(key=key)
    expected = env.step(state, action)
    result = jax.jit(env.step)(state, action)
    assert_trees_close(expected, result)


def test_vmap_reset(env):
    """jax.vmap over reset produces batched outputs."""
    keys = jax.random.split(jax.random.key(0), 4)
    states = jax.vmap(env.reset)(key=keys)
    for leaf in jtu.tree_leaves(states):
        assert leaf.shape[0] == 4


def test_vmap_step(env, key):
    """jax.vmap over step processes a batch."""
    keys = jax.random.split(key, 4)
    states = jax.vmap(env.reset)(key=keys)
    actions = jax.vmap(lambda k: env.action_space.sample(key=k))(keys)
    new_states = jax.vmap(env.step)(states, actions)
    for leaf in jtu.tree_leaves(new_states):
        assert leaf.shape[0] == 4


def test_pytree_roundtrip(env, key):
    """Flatten then unflatten reconstructs State exactly."""
    state = env.reset(key=key)
    leaves, treedef = jtu.tree_flatten(state)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    assert_trees_equal(state, reconstructed)


def test_has_spaces(env):
    """Env exposes action_space and observation_space with sample methods."""
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert hasattr(env.action_space, "sample")
    assert hasattr(env.observation_space, "sample")


def test_reward_dtype(env, key):
    """Reward is a floating-point scalar."""
    state = env.reset(key=key)
    state = env.step(state, env.action_space.sample(key=key))
    assert jnp.issubdtype(state.reward.dtype, jnp.floating)


def test_termination_dtype(env, key):
    """Termination and truncation are boolean."""
    state = env.reset(key=key)
    assert state.termination.dtype == jnp.bool_
    assert state.truncation.dtype == jnp.bool_
