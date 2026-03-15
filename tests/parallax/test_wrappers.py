"""Tests for VmapWrapper and TimeLimit wrappers."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from conftest import assert_trees_close, assert_trees_equal
from parallax.spaces import Box, Discrete
from parallax.struct import EnvState, Timestep
from parallax.wrappers import TimeLimit, VmapWrapper


class _ScalarEnv:
    """Env with scalar state."""

    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=100.0, shape=())

    def reset(self, *, key):
        state = EnvState(state=jnp.float32(0.0), step=jnp.int32(0), key=key)
        ts = Timestep(
            observation=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        return state, ts

    def step(self, state, action):
        count = state.state + 1.0
        new_state = EnvState(state=count, step=state.step + 1, key=state.key)
        ts = Timestep(
            observation=count,
            reward=jnp.float32(1.0),
            termination=count >= 10.0,
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        return new_state, ts


class _VectorEnv:
    """Env with vector state (exercises jnp.where broadcasting)."""

    action_space = Discrete(4)
    observation_space = Box(low=0.0, high=4.0, shape=(2,))

    def reset(self, *, key):
        pos = jnp.zeros(2, dtype=jnp.float32)
        state = EnvState(state=pos, step=jnp.int32(0), key=key)
        ts = Timestep(
            observation=pos,
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        return state, ts

    def step(self, state, action):
        moves = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.float32)
        new_pos = jnp.clip(state.state + moves[action], 0.0, 4.0)
        new_state = EnvState(state=new_pos, step=state.step + 1, key=state.key)
        at_goal = jnp.all(new_pos == jnp.array([4.0, 4.0]))
        ts = Timestep(
            observation=new_pos,
            reward=jnp.where(at_goal, 1.0, 0.0),
            termination=at_goal,
            truncation=jnp.bool_(False),
            info=jnp.zeros(()),
        )
        return new_state, ts


class TestVmapWrapper:
    def test_batched_reset(self):
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, timesteps = env.reset(key=keys)
        for leaf in jtu.tree_leaves(states):
            assert leaf.shape[0] == 4
        for leaf in jtu.tree_leaves(timesteps):
            assert leaf.shape[0] == 4

    def test_batched_step(self):
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, _ = env.reset(key=keys)
        actions = jnp.zeros(4, dtype=jnp.int32)
        new_states, timesteps = env.step(states, actions)
        for leaf in jtu.tree_leaves(new_states):
            assert leaf.shape[0] == 4

    def test_selective_reset_scalar_state(self):
        """Selective reset only resets envs where done=True (scalar state)."""
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, timesteps = env.reset(key=keys)

        actions = jnp.zeros(4, dtype=jnp.int32)
        states, timesteps = env.step(states, actions)

        done = jnp.array([True, False, True, False])
        new_states, _ = env.reset(
            key=jax.random.key(1), state=states, timestep=timesteps, done=done
        )

        assert new_states.step[0] == 0  # reset
        assert new_states.step[1] == 1  # kept
        assert new_states.step[2] == 0  # reset
        assert new_states.step[3] == 1  # kept

    def test_selective_reset_vector_state(self):
        """Selective reset with vector-valued state (tests jnp.where broadcasting)."""
        env = VmapWrapper(_VectorEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, timesteps = env.reset(key=keys)

        actions = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        states, timesteps = env.step(states, actions)

        done = jnp.array([True, False, False, True])
        new_states, _ = env.reset(
            key=jax.random.key(1), state=states, timestep=timesteps, done=done
        )

        assert jnp.allclose(new_states.state[0], jnp.zeros(2))
        assert jnp.allclose(new_states.state[3], jnp.zeros(2))
        assert new_states.step[1] == 1
        assert new_states.step[2] == 1

    def test_selective_reset_preserves_timestep(self):
        """Selective reset only updates timestep fields for done envs."""
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, timesteps = env.reset(key=keys)

        actions = jnp.zeros(4, dtype=jnp.int32)
        states, timesteps = env.step(states, actions)
        old_rewards = timesteps.reward.copy()

        done = jnp.array([True, False, False, False])
        _, new_timesteps = env.reset(
            key=jax.random.key(1), state=states, timestep=timesteps, done=done
        )

        assert new_timesteps.reward[0] == 0.0
        assert jnp.array_equal(new_timesteps.reward[1:], old_rewards[1:])

    def test_forwards_spaces(self):
        base = _ScalarEnv()
        env = VmapWrapper(base)
        assert env.action_space is base.action_space
        assert env.observation_space is base.observation_space

    def test_jit_reset(self):
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        s1, ts1 = env.reset(key=keys)
        s2, ts2 = jax.jit(env.reset)(key=keys)
        assert_trees_close(s1, s2)
        assert_trees_close(ts1, ts2)

    def test_jit_step(self):
        env = VmapWrapper(_ScalarEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, _ = env.reset(key=keys)
        actions = jnp.zeros(4, dtype=jnp.int32)
        s1, ts1 = env.step(states, actions)
        s2, ts2 = jax.jit(env.step)(states, actions)
        assert_trees_close(s1, s2)
        assert_trees_close(ts1, ts2)

    def test_jit_selective_reset(self):
        """Selective reset works under jax.jit."""
        env = VmapWrapper(_VectorEnv())
        keys = jax.random.split(jax.random.key(0), 4)
        states, timesteps = env.reset(key=keys)
        actions = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        states, timesteps = env.step(states, actions)

        done = jnp.array([True, False, False, True])
        reset_key = jax.random.key(1)
        new_states, _ = jax.jit(
            lambda k: env.reset(key=k, state=states, timestep=timesteps, done=done)
        )(reset_key)
        assert new_states.step[0] == 0
        assert new_states.step[1] == 1


class TestTimeLimit:
    def test_no_early_truncation(self):
        """Steps before max_steps should not be truncated."""
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state, _ = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(4):
            state, ts = env.step(state, action)
            assert not ts.truncation

    def test_truncation_at_limit(self):
        """Step at max_steps triggers truncation."""
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state, ts = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(5):
            state, ts = env.step(state, action)
        assert ts.truncation

    def test_truncation_after_limit(self):
        """Steps beyond max_steps also truncate."""
        env = TimeLimit(_ScalarEnv(), max_steps=3)
        state, ts = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(5):
            state, ts = env.step(state, action)
        assert ts.truncation

    def test_preserves_termination(self):
        """TimeLimit does not override the base env's termination signal."""
        env = TimeLimit(_ScalarEnv(), max_steps=100)
        state, ts = env.reset(key=jax.random.key(0))
        action = jnp.int32(0)
        for _ in range(10):
            state, ts = env.step(state, action)
        assert ts.termination
        assert not ts.truncation

    def test_truncation_or_semantics(self):
        """Truncation uses OR — existing truncation from base env is preserved."""
        env = TimeLimit(_ScalarEnv(), max_steps=50)
        state, _ = env.reset(key=jax.random.key(0))
        state, ts = env.step(state, jnp.int32(0))
        assert not ts.truncation
        for _ in range(49):
            state, ts = env.step(state, jnp.int32(0))
        assert ts.truncation

    def test_reset_delegates(self):
        """TimeLimit.reset produces same output as base env.reset."""
        base = _ScalarEnv()
        env = TimeLimit(base, max_steps=10)
        key = jax.random.key(0)
        base_s, base_ts = base.reset(key=key)
        env_s, env_ts = env.reset(key=key)
        assert_trees_equal(base_s, env_s)
        assert_trees_equal(base_ts, env_ts)

    def test_forwards_spaces(self):
        base = _ScalarEnv()
        env = TimeLimit(base, max_steps=10)
        assert env.action_space is base.action_space
        assert env.observation_space is base.observation_space

    def test_jit_compatible(self):
        env = TimeLimit(_ScalarEnv(), max_steps=5)
        state, _ = jax.jit(env.reset)(key=jax.random.key(0))
        state, ts = jax.jit(env.step)(state, jnp.int32(0))
        assert state.step == 1
        assert not ts.truncation

    def test_composable_with_vmap(self):
        """TimeLimit + VmapWrapper compose correctly."""
        env = VmapWrapper(TimeLimit(_ScalarEnv(), max_steps=3))
        keys = jax.random.split(jax.random.key(0), 4)
        states, ts = env.reset(key=keys)
        actions = jnp.zeros(4, dtype=jnp.int32)
        for _ in range(3):
            states, ts = env.step(states, actions)
        assert jnp.all(ts.truncation)
