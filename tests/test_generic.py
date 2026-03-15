"""Cross-environment generic tests.

Every test here runs once per env registered in conftest (CountingEnv, GridWorldEnv).
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from conftest import assert_trees_close, assert_trees_equal
from parallax.struct import EnvState, Timestep


def test_reset_returns_state_and_timestep(env, key):
    """reset returns (EnvState, Timestep)."""
    state, timestep = env.reset(key=key)
    assert isinstance(state, EnvState)
    assert isinstance(timestep, Timestep)


def test_step_returns_state_and_timestep(env, key):
    """step returns (EnvState, Timestep)."""
    state, _ = env.reset(key=key)
    action = env.action_space.sample(key=key)
    new_state, timestep = env.step(state, action)
    assert isinstance(new_state, EnvState)
    assert isinstance(timestep, Timestep)


def test_step_increments(env, key):
    """step count increments each step."""
    state, _ = env.reset(key=key)
    assert state.step == 0
    action = env.action_space.sample(key=key)
    state, _ = env.step(state, action)
    assert state.step == 1


def test_reset_not_done(env, key):
    """Initial timestep is neither terminal nor truncated."""
    _, timestep = env.reset(key=key)
    assert not timestep.termination
    assert not timestep.truncation


def test_reset_determinism(env, key):
    """Same key produces identical reset outputs."""
    s1, ts1 = env.reset(key=key)
    s2, ts2 = env.reset(key=key)
    assert_trees_equal(s1, s2)
    assert_trees_equal(ts1, ts2)


def test_step_determinism(env, key):
    """Same state and action produce identical step outputs."""
    state, _ = env.reset(key=key)
    action = env.action_space.sample(key=key)
    s1, ts1 = env.step(state, action)
    s2, ts2 = env.step(state, action)
    assert_trees_equal(s1, s2)
    assert_trees_equal(ts1, ts2)


def test_jit_reset(env, key):
    """jax.jit produces the same reset output as eager execution."""
    expected_s, expected_ts = env.reset(key=key)
    s, ts = jax.jit(env.reset)(key=key)
    assert_trees_close(expected_s, s)
    assert_trees_close(expected_ts, ts)


def test_jit_step(env, key):
    """jax.jit produces the same step output as eager execution."""
    state, _ = env.reset(key=key)
    action = env.action_space.sample(key=key)
    expected_s, expected_ts = env.step(state, action)
    s, ts = jax.jit(env.step)(state, action)
    assert_trees_close(expected_s, s)
    assert_trees_close(expected_ts, ts)


def test_vmap_reset(env):
    """jax.vmap over reset produces batched outputs."""
    keys = jax.random.split(jax.random.key(0), 4)
    states, timesteps = jax.vmap(env.reset)(key=keys)
    for leaf in jtu.tree_leaves(states):
        assert leaf.shape[0] == 4
    for leaf in jtu.tree_leaves(timesteps):
        assert leaf.shape[0] == 4


def test_vmap_step(env, key):
    """jax.vmap over step processes a batch."""
    keys = jax.random.split(key, 4)
    states, _ = jax.vmap(env.reset)(key=keys)
    actions = jax.vmap(lambda k: env.action_space.sample(key=k))(keys)
    new_states, timesteps = jax.vmap(env.step)(states, actions)
    for leaf in jtu.tree_leaves(new_states):
        assert leaf.shape[0] == 4
    for leaf in jtu.tree_leaves(timesteps):
        assert leaf.shape[0] == 4


def test_pytree_roundtrip_state(env, key):
    """Flatten then unflatten reconstructs EnvState exactly."""
    state, _ = env.reset(key=key)
    leaves, treedef = jtu.tree_flatten(state)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    assert_trees_equal(state, reconstructed)


def test_pytree_roundtrip_timestep(env, key):
    """Flatten then unflatten reconstructs Timestep exactly."""
    _, timestep = env.reset(key=key)
    leaves, treedef = jtu.tree_flatten(timestep)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    assert_trees_equal(timestep, reconstructed)


def test_has_spaces(env):
    """Env exposes action_space and observation_space with sample methods."""
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert hasattr(env.action_space, "sample")
    assert hasattr(env.observation_space, "sample")


def test_reward_dtype(env, key):
    """Reward is a floating-point scalar."""
    _, timestep = env.reset(key=key)
    assert jnp.issubdtype(timestep.reward.dtype, jnp.floating)


def test_termination_dtype(env, key):
    """Termination and truncation are boolean."""
    _, timestep = env.reset(key=key)
    assert timestep.termination.dtype == jnp.bool_
    assert timestep.truncation.dtype == jnp.bool_
