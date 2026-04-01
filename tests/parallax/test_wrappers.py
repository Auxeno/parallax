"""Tests for VmapWrapper, AutoResetWrapper, and TimeLimit wrappers."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from conftest import assert_trees_close, assert_trees_equal
from parallax.core import State
from parallax.spaces import Box, Discrete
from parallax.wrappers import AutoResetWrapper, TimeLimit, VmapWrapper


class _ScalarEnv:
    """Env with scalar state."""

    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=100.0, shape=())

    def reset(self, *, key):
        return State(
            env_state=jnp.float32(0.0),
            observation=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state, action):
        env_state = state.env_state + 1.0
        step_count = state.step_count + 1
        return State(
            env_state=env_state,
            observation=env_state.astype(jnp.float32),
            reward=jnp.where(step_count > 0, jnp.float32(1.0), jnp.float32(0.0)),
            termination=env_state >= 10.0,
            truncation=jnp.bool_(False),
            info={},
            step_count=step_count,
            key=state.key,
        )


class _VectorEnv:
    """Env with vector state (exercises jnp.where broadcasting)."""

    action_space = Discrete(4)
    observation_space = Box(low=0.0, high=4.0, shape=(2,))

    def reset(self, *, key):
        pos = jnp.zeros(2, dtype=jnp.float32)
        return State(
            env_state=pos,
            observation=pos,
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state, action):
        moves = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.float32)
        pos = jnp.clip(state.env_state + moves[action], 0.0, 4.0)
        at_goal = jnp.all(pos == jnp.array([4.0, 4.0]))
        return State(
            env_state=pos,
            observation=pos,
            reward=jnp.where(at_goal, 1.0, 0.0),
            termination=at_goal,
            truncation=jnp.bool_(False),
            info={},
            step_count=state.step_count + 1,
            key=state.key,
        )


class TestVmapWrapper:
    def test_batched_reset(self):
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        for leaf in jtu.tree_leaves(state):
            assert leaf.shape[0] == 4

    def test_batched_step(self):
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.zeros(4, dtype=jnp.int32)
        new_state = env.step(state, actions)
        for leaf in jtu.tree_leaves(new_state):
            assert leaf.shape[0] == 4

    def test_selective_reset_scalar_state(self):
        """Selective reset only resets envs where done=True (scalar state)."""
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jnp.zeros(4, dtype=jnp.int32)
        state = env.step(state, actions)

        done = jnp.array([True, False, True, False])
        new_state = env.reset(key=jax.random.key(1), state=state, done=done)

        assert new_state.step_count[0] == 0  # reset
        assert new_state.step_count[1] == 1  # kept
        assert new_state.step_count[2] == 0  # reset
        assert new_state.step_count[3] == 1  # kept

    def test_selective_reset_vector_state(self):
        """Selective reset with vector-valued state (tests jnp.where broadcasting)."""
        env = VmapWrapper(_VectorEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        state = env.step(state, actions)

        done = jnp.array([True, False, False, True])
        new_state = env.reset(key=jax.random.key(1), state=state, done=done)

        assert jnp.allclose(new_state.env_state[0], jnp.zeros(2))
        assert jnp.allclose(new_state.env_state[3], jnp.zeros(2))
        assert new_state.step_count[1] == 1
        assert new_state.step_count[2] == 1

    def test_selective_reset_preserves_non_done(self):
        """Selective reset only updates fields for done envs."""
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jnp.zeros(4, dtype=jnp.int32)
        state = env.step(state, actions)
        old_state_values = state.env_state.copy()

        done = jnp.array([True, False, False, False])
        new_state = env.reset(key=jax.random.key(1), state=state, done=done)

        assert new_state.env_state[0] == 0.0  # reset
        assert jnp.array_equal(new_state.env_state[1:], old_state_values[1:])

    def test_forwards_spaces(self):
        base = _ScalarEnv()
        env = VmapWrapper(base, num_envs=4)
        assert env.action_space is base.action_space
        assert env.observation_space is base.observation_space

    def test_jit_reset(self):
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        key = jax.random.key(0)
        s1 = env.reset(key=key)
        s2 = jax.jit(env.reset)(key=key)
        assert_trees_close(s1, s2)

    def test_jit_step(self):
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.zeros(4, dtype=jnp.int32)
        s1 = env.step(state, actions)
        s2 = jax.jit(env.step)(state, actions)
        assert_trees_close(s1, s2)

    def test_jit_selective_reset(self):
        """Selective reset works under jax.jit."""
        env = VmapWrapper(_VectorEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        state = env.step(state, actions)

        done = jnp.array([True, False, False, True])
        reset_key = jax.random.key(1)
        new_state = jax.jit(lambda k: env.reset(key=k, state=state, done=done))(reset_key)
        assert new_state.step_count[0] == 0
        assert new_state.step_count[1] == 1

    def test_properties_batched(self):
        """Precomputed properties are batched after VmapWrapper."""
        env = VmapWrapper(_ScalarEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        state = env.step(state, jnp.zeros(4, dtype=jnp.int32))
        assert state.observation.shape == (4,)
        assert state.reward.shape == (4,)
        assert state.termination.shape == (4,)


class TestAutoResetWrapper:
    def test_resets_on_termination(self):
        """State is reset when the episode terminates."""
        env = AutoResetWrapper(_ScalarEnv())
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(10):
            state = env.step(state, action)
        assert state.step_count == 0
        assert not state.termination

    def test_no_reset_before_done(self):
        """State progresses normally when the episode is not done."""
        env = AutoResetWrapper(_ScalarEnv())
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for i in range(1, 5):
            state = env.step(state, action)
            assert state.step_count == i
            assert not state.termination

    def test_resets_on_truncation(self):
        """State is reset when the episode is truncated."""
        env = AutoResetWrapper(TimeLimit(_ScalarEnv(), max_steps=3))
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(3):
            state = env.step(state, action)
        assert state.step_count == 0
        assert not state.truncation

    def test_continues_after_reset(self):
        """Environment keeps running across episode boundaries."""
        env = AutoResetWrapper(_ScalarEnv())
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(10):
            state = env.step(state, action)
        assert state.step_count == 0
        state = env.step(state, action)
        assert state.step_count == 1

    def test_forwards_spaces(self):
        base = _ScalarEnv()
        env = AutoResetWrapper(base)
        assert env.action_space is base.action_space
        assert env.observation_space is base.observation_space

    def test_jit_compatible(self):
        env = AutoResetWrapper(_ScalarEnv())
        state = jax.jit(env.reset)(key=jax.random.key(0))
        s1 = env.step(state, jnp.int32(0))
        s2 = jax.jit(env.step)(state, jnp.int32(0))
        assert_trees_close(s1, s2)

    def test_composable_with_vmap(self):
        """AutoResetWrapper + VmapWrapper compose correctly."""
        env = VmapWrapper(AutoResetWrapper(_ScalarEnv()), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.zeros(4, dtype=jnp.int32)
        for _ in range(10):
            state = env.step(state, actions)
        assert jnp.all(state.step_count == 0)

    def test_composable_with_timelimit(self):
        """AutoResetWrapper + TimeLimit + VmapWrapper compose correctly."""
        env = VmapWrapper(AutoResetWrapper(TimeLimit(_ScalarEnv(), max_steps=3)), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.zeros(4, dtype=jnp.int32)
        for _ in range(3):
            state = env.step(state, actions)
        assert jnp.all(state.step_count == 0)

    def test_key_advances_each_step(self):
        """RNG key changes every step, even without episode termination."""
        env = AutoResetWrapper(_ScalarEnv())
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        keys_seen = [jax.random.key_data(state.key)]
        for _ in range(5):
            state = env.step(state, action)
            keys_seen.append(jax.random.key_data(state.key))
        for i in range(len(keys_seen) - 1):
            assert not jnp.array_equal(keys_seen[i], keys_seen[i + 1])

    def test_key_advances_through_reset(self):
        """RNG key keeps advancing across episode boundaries."""
        env = AutoResetWrapper(_ScalarEnv())
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        keys_seen = []
        for _ in range(12):  # terminates at step 10, then 2 more
            state = env.step(state, action)
            keys_seen.append(jax.random.key_data(state.key))
        # All keys should be unique
        for i in range(len(keys_seen)):
            for j in range(i + 1, len(keys_seen)):
                assert not jnp.array_equal(keys_seen[i], keys_seen[j])

    def test_reset_delegates(self):
        """AutoResetWrapper.reset produces same leaf values as base env.reset."""
        base = _ScalarEnv()
        env = AutoResetWrapper(base)
        key = jax.random.key(0)
        base_s = base.reset(key=key)
        env_s = env.reset(key=key)
        assert_trees_equal(base_s, env_s)


class TestTimeLimit:
    def test_no_early_truncation(self):
        """Steps before max_steps should not be truncated."""
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(4):
            state = env.step(state, action)
            assert not state.truncation

    def test_truncation_at_limit(self):
        """Step at max_steps triggers truncation."""
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(5):
            state = env.step(state, action)
        assert state.truncation

    def test_truncation_after_limit(self):
        """Steps beyond max_steps also truncate."""
        env = TimeLimit(_ScalarEnv(), max_steps=3)
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(5):
            state = env.step(state, action)
        assert state.truncation

    def test_preserves_termination(self):
        """TimeLimit does not override the base env's termination signal."""
        env = TimeLimit(_ScalarEnv(), max_steps=100)
        state = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(10):
            state = env.step(state, action)
        assert state.termination
        assert not state.truncation

    def test_truncation_or_semantics(self):
        """Truncation uses OR: existing truncation from base env is preserved."""
        env = TimeLimit(_ScalarEnv(), max_steps=50)
        state = env.reset(key=jax.random.key(0))
        state = env.step(state, jnp.int32(0))
        assert not state.truncation
        for _ in range(49):
            state = env.step(state, jnp.int32(0))
        assert state.truncation

    def test_reset_delegates(self):
        """TimeLimit.reset produces same leaf values as base env.reset."""
        base = _ScalarEnv()
        env = TimeLimit(base, max_steps=10)
        key = jax.random.key(0)
        base_s = base.reset(key=key)
        env_s = env.reset(key=key)
        assert_trees_equal(base_s, env_s)

    def test_forwards_spaces(self):
        base = _ScalarEnv()
        env = TimeLimit(base, max_steps=10)
        assert env.action_space is base.action_space
        assert env.observation_space is base.observation_space

    def test_jit_compatible(self):
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state = jax.jit(env.reset)(key=jax.random.key(0))
        state = jax.jit(env.step)(state, jnp.int32(0))
        assert state.step_count == 1
        assert not state.truncation

    def test_composable_with_vmap(self):
        """TimeLimit + VmapWrapper compose correctly."""
        env = VmapWrapper(TimeLimit(_ScalarEnv(), max_steps=3), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.zeros(4, dtype=jnp.int32)
        for _ in range(3):
            state = env.step(state, actions)
        assert jnp.all(state.truncation)


class TestSubclassedState:
    """Subclassed State with extra fields works through wrappers."""

    @jax.tree_util.register_dataclass
    @dataclass
    class _MaskedState(State):
        action_mask: jax.Array

    class _MaskedEnv(_ScalarEnv):
        def reset(self, *, key):
            s = _ScalarEnv.reset(self, key=key)
            return TestSubclassedState._MaskedState(
                **{f.name: getattr(s, f.name) for f in s.__dataclass_fields__.values()},
                action_mask=jnp.array([True, True]),
            )

        def step(self, state, action):
            s = _ScalarEnv.step(self, state, action)
            return TestSubclassedState._MaskedState(
                **{f.name: getattr(s, f.name) for f in s.__dataclass_fields__.values()},
                action_mask=jnp.array([True, s.env_state < 5.0]),
            )

    def test_through_vmap(self):
        env = VmapWrapper(self._MaskedEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        state = env.step(state, jnp.zeros(4, dtype=jnp.int32))
        assert state.action_mask.shape == (4, 2)  # type: ignore[attr-defined]

    def test_through_wrapper_stack(self):
        env = VmapWrapper(TimeLimit(self._MaskedEnv(), max_steps=10), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        state = env.step(state, jnp.zeros(4, dtype=jnp.int32))
        assert state.action_mask.shape == (4, 2)  # type: ignore[attr-defined]
