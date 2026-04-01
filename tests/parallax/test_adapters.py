"""Tests for environment adapters."""

import warnings

import jax
import jax.numpy as jnp

from parallax.adapters.brax import BraxAdapter
from parallax.adapters.gymnax import GymnaxAdapter
from parallax.adapters.mjx import MJXAdapter
from parallax.core import State
from parallax.wrappers import AutoResetWrapper, TimeLimit, VmapWrapper

gymnax = __import__("pytest").importorskip("gymnax")
brax_envs = __import__("pytest").importorskip("brax.envs")
mujoco_playground = __import__("pytest").importorskip("mujoco_playground")


def _make_env():
    env, _ = gymnax.make("CartPole-v1")
    return GymnaxAdapter(env)


class TestGymnaxAdapter:
    def test_reset_returns_state(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        assert isinstance(state, State)

    def test_reset_observation_shape(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (4,)

    def test_reset_not_done(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        assert not state.termination
        assert not state.truncation
        assert not state.done

    def test_step_returns_state(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert isinstance(state, State)
        assert state.step_count == 1

    def test_step_reward_dtype(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert jnp.issubdtype(state.reward.dtype, jnp.floating)

    def test_step_advances_key(self):
        env = _make_env()
        state = env.reset(key=jax.random.key(0))
        key_before = jax.random.key_data(state.key)
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        key_after = jax.random.key_data(state.key)
        assert not jnp.array_equal(key_before, key_after)

    def test_spaces(self):
        env = _make_env()
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env.action_space, "sample")
        assert hasattr(env.observation_space, "sample")

    def test_with_vmap_wrapper(self):
        env = VmapWrapper(_make_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (4, 4)
        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)
        assert state.observation.shape == (4, 4)
        assert state.reward.shape == (4,)

    def test_terminal_observation_preserved(self):
        """Terminal obs is the true final observation, not a reset observation.

        This verifies no hidden auto-reset is occurring.
        """
        env = _make_env()
        state = env.reset(key=jax.random.key(0))

        # Step until termination
        for _ in range(500):
            action = env.action_space.sample(key=state.key)
            state = env.step(state, action)
            if state.done:
                break

        assert state.done, "CartPole should terminate within 500 steps"
        terminal_obs = state.observation

        # The terminal observation should be a real state, not zeros or a reset
        assert terminal_obs.shape == (4,)
        assert not jnp.all(terminal_obs == 0.0)

        # Step once more from the terminal state without resetting
        action = env.action_space.sample(key=state.key)
        next_state = env.step(state, action)

        # Observation should have changed (env keeps evolving, not stuck)
        assert not jnp.array_equal(terminal_obs, next_state.observation)

    def test_with_autoreset(self):
        """AutoResetWrapper resets the env on termination."""
        env = AutoResetWrapper(_make_env())
        state = env.reset(key=jax.random.key(0))

        for _ in range(500):
            action = env.action_space.sample(key=state.key)
            state = env.step(state, action)
            if state.step_count == 0:
                # Auto-reset happened, env should not be done
                assert not state.done
                break
        else:
            raise AssertionError("AutoReset should have triggered within 500 steps")

    def test_with_timelimit(self):
        """TimeLimit truncates the episode."""
        env = TimeLimit(_make_env(), max_steps=5)
        state = env.reset(key=jax.random.key(0))
        for _ in range(5):
            action = env.action_space.sample(key=state.key)
            state = env.step(state, action)
        assert state.truncation
        assert state.done

    def test_vmap_selective_reset(self):
        """VmapWrapper selective reset works with gymnax adapter."""
        env = VmapWrapper(_make_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)

        done = jnp.array([True, False, False, True])
        new_state = env.reset(key=jax.random.key(2), state=state, done=done)

        assert new_state.step_count[0] == 0  # reset
        assert new_state.step_count[1] == 1  # kept
        assert new_state.step_count[2] == 1  # kept
        assert new_state.step_count[3] == 0  # reset

    def test_full_wrapper_stack(self):
        """Full stack: TimeLimit + AutoReset + Vmap composes correctly."""
        env = VmapWrapper(AutoResetWrapper(TimeLimit(_make_env(), max_steps=10)), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        for _ in range(20):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        # After 20 steps with max_steps=10, all envs should have auto-reset
        assert jnp.all(state.step_count < 10)

    def test_vmap_terminal_obs(self):
        """Terminal observations are correct in batched setting."""
        env = VmapWrapper(TimeLimit(_make_env(), max_steps=5), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        # Step to truncation
        for _ in range(5):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        # All envs should be truncated at step 5
        assert jnp.all(state.truncation)
        assert jnp.all(state.step_count == 5)

        # Terminal observations should be real, not zeros
        assert not jnp.all(state.observation == 0.0)

        # Now manually reset and verify obs changed
        new_state = env.reset(key=jax.random.key(1), state=state, done=state.done)
        assert jnp.all(new_state.step_count == 0)
        assert not jnp.array_equal(state.observation, new_state.observation)


def _make_brax_env():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = brax_envs.get_environment("inverted_pendulum")
    return BraxAdapter(raw)


class TestBraxAdapter:
    def test_reset_returns_state(self):
        env = _make_brax_env()
        state = env.reset(key=jax.random.key(0))
        assert isinstance(state, State)

    def test_reset_observation_shape(self):
        env = _make_brax_env()
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (4,)

    def test_reset_not_done(self):
        env = _make_brax_env()
        state = env.reset(key=jax.random.key(0))
        assert not state.termination
        assert not state.truncation
        assert not state.done

    def test_step_returns_state(self):
        env = _make_brax_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert isinstance(state, State)
        assert state.step_count == 1

    def test_step_reward_dtype(self):
        env = _make_brax_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert jnp.issubdtype(state.reward.dtype, jnp.floating)

    def test_spaces(self):
        env = _make_brax_env()
        assert hasattr(env.action_space, "sample")
        assert hasattr(env.observation_space, "sample")

    def test_with_vmap_wrapper(self):
        env = VmapWrapper(_make_brax_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (4, 4)
        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)
        assert state.observation.shape == (4, 4)
        assert state.reward.shape == (4,)

    def test_with_timelimit(self):
        """TimeLimit truncates the episode."""
        env = TimeLimit(_make_brax_env(), max_steps=5)
        state = env.reset(key=jax.random.key(0))
        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)
        assert state.truncation
        assert state.done

    def test_with_autoreset(self):
        """AutoResetWrapper resets on truncation via TimeLimit."""
        env = AutoResetWrapper(TimeLimit(_make_brax_env(), max_steps=5))
        state = env.reset(key=jax.random.key(0))
        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)
        # Should have auto-reset
        assert state.step_count == 0
        assert not state.done

    def test_vmap_selective_reset(self):
        """VmapWrapper selective reset works with brax adapter."""
        env = VmapWrapper(_make_brax_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)

        done = jnp.array([True, False, False, True])
        new_state = env.reset(key=jax.random.key(2), state=state, done=done)

        assert new_state.step_count[0] == 0  # reset
        assert new_state.step_count[1] == 1  # kept
        assert new_state.step_count[2] == 1  # kept
        assert new_state.step_count[3] == 0  # reset

    def test_full_wrapper_stack(self):
        """Full stack: TimeLimit + AutoReset + Vmap composes correctly."""
        env = VmapWrapper(AutoResetWrapper(TimeLimit(_make_brax_env(), max_steps=10)), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        for _ in range(20):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        # After 20 steps with max_steps=10, all envs should have auto-reset
        assert jnp.all(state.step_count < 10)

    def test_terminal_observation_preserved(self):
        """Terminal obs via TimeLimit is the true final observation."""
        env = TimeLimit(_make_brax_env(), max_steps=5)
        state = env.reset(key=jax.random.key(0))

        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)

        assert state.done
        terminal_obs = state.observation
        assert terminal_obs.shape == (4,)
        assert not jnp.all(terminal_obs == 0.0)

    def test_vmap_terminal_obs(self):
        """Terminal observations are correct in batched setting."""
        env = VmapWrapper(TimeLimit(_make_brax_env(), max_steps=5), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        for _ in range(5):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        assert jnp.all(state.truncation)
        assert not jnp.all(state.observation == 0.0)

        new_state = env.reset(key=jax.random.key(1), state=state, done=state.done)
        assert jnp.all(new_state.step_count == 0)

    def test_strips_brax_wrappers(self):
        """BraxAdapter strips Brax wrappers and extracts episode_length."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brax_env = brax_envs.create("inverted_pendulum", episode_length=42)
        env = BraxAdapter(brax_env)
        assert env.episode_length == 42

    def test_brax_create_with_timelimit(self):
        """BraxAdapter truncates at the extracted episode_length."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brax_env = brax_envs.create("halfcheetah", episode_length=5)
        env = BraxAdapter(brax_env)

        state = env.reset(key=jax.random.key(0))
        assert not state.done

        for i in range(5):
            action = env.action_space.sample(key=jax.random.key(i))
            state = env.step(state, action)

        # Should be truncated, not terminated
        assert state.truncation
        assert not state.termination
        assert state.done
        assert state.step_count == 5

        # No autoreset - stepping past the limit continues
        action = env.action_space.sample(key=jax.random.key(99))
        next_state = env.step(state, action)
        assert next_state.step_count == 6


def _make_mjx_env():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mujoco_playground.registry.load("CartpoleBalance", config_overrides={"impl": "jax"})
    return MJXAdapter(raw)


class TestMJXAdapter:
    def test_reset_returns_state(self):
        env = _make_mjx_env()
        state = env.reset(key=jax.random.key(0))
        assert isinstance(state, State)

    def test_reset_observation_shape(self):
        env = _make_mjx_env()
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (5,)

    def test_reset_not_done(self):
        env = _make_mjx_env()
        state = env.reset(key=jax.random.key(0))
        assert not state.termination
        assert not state.truncation
        assert not state.done

    def test_step_returns_state(self):
        env = _make_mjx_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert isinstance(state, State)
        assert state.step_count == 1

    def test_step_reward_dtype(self):
        env = _make_mjx_env()
        state = env.reset(key=jax.random.key(0))
        action = env.action_space.sample(key=jax.random.key(1))
        state = env.step(state, action)
        assert jnp.issubdtype(state.reward.dtype, jnp.floating)

    def test_spaces(self):
        env = _make_mjx_env()
        assert hasattr(env.action_space, "sample")
        assert hasattr(env.observation_space, "sample")

    def test_with_vmap_wrapper(self):
        env = VmapWrapper(_make_mjx_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        assert state.observation.shape == (4, 5)
        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)
        assert state.observation.shape == (4, 5)
        assert state.reward.shape == (4,)

    def test_with_timelimit(self):
        """TimeLimit truncates the episode."""
        env = TimeLimit(_make_mjx_env(), max_steps=5)
        state = env.reset(key=jax.random.key(0))
        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)
        assert state.truncation
        assert state.done

    def test_with_autoreset(self):
        """AutoResetWrapper resets on truncation via TimeLimit."""
        env = AutoResetWrapper(TimeLimit(_make_mjx_env(), max_steps=5))
        state = env.reset(key=jax.random.key(0))
        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)
        assert state.step_count == 0
        assert not state.done

    def test_vmap_selective_reset(self):
        """VmapWrapper selective reset works with MJX adapter."""
        env = VmapWrapper(_make_mjx_env(), num_envs=4)
        state = env.reset(key=jax.random.key(0))

        actions = jax.vmap(env.action_space.sample)(key=jax.random.split(jax.random.key(1), 4))
        state = env.step(state, actions)

        done = jnp.array([True, False, False, True])
        new_state = env.reset(key=jax.random.key(2), state=state, done=done)

        assert new_state.step_count[0] == 0  # reset
        assert new_state.step_count[1] == 1  # kept
        assert new_state.step_count[2] == 1  # kept
        assert new_state.step_count[3] == 0  # reset

    def test_full_wrapper_stack(self):
        """Full stack: TimeLimit + AutoReset + Vmap composes correctly."""
        env = VmapWrapper(AutoResetWrapper(TimeLimit(_make_mjx_env(), max_steps=10)), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        for _ in range(20):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        assert jnp.all(state.step_count < 10)

    def test_terminal_observation_preserved(self):
        """Terminal obs via TimeLimit is the true final observation."""
        env = TimeLimit(_make_mjx_env(), max_steps=5)
        state = env.reset(key=jax.random.key(0))

        for _ in range(5):
            action = env.action_space.sample(key=jax.random.key(1))
            state = env.step(state, action)

        assert state.done
        terminal_obs = state.observation
        assert terminal_obs.shape == (5,)
        assert not jnp.all(terminal_obs == 0.0)

    def test_vmap_terminal_obs(self):
        """Terminal observations are correct in batched setting."""
        env = VmapWrapper(TimeLimit(_make_mjx_env(), max_steps=5), num_envs=4)
        key = jax.random.key(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(key=reset_key)

        for _ in range(5):
            key, action_key = jax.random.split(key)
            actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, 4))
            state = env.step(state, actions)

        assert jnp.all(state.truncation)
        assert not jnp.all(state.observation == 0.0)

        new_state = env.reset(key=jax.random.key(1), state=state, done=state.done)
        assert jnp.all(new_state.step_count == 0)
