"""Static typing contract tests."""

from dataclasses import dataclass, replace
from typing import assert_type

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from parallax.core import Agents, Env, MARLEnv, MARLState, MARLVectorEnv, State, VectorEnv
from parallax.spaces import Box, Discrete, Space, stack_space
from parallax.wrappers import TimeLimit, VmapWrapper


class _CounterEnv:
    """Fully annotated single-agent env, actions typed as Array as env authors write them."""

    @property
    def action_space(self) -> Space:
        return Discrete(2)

    @property
    def observation_space(self) -> Space:
        return Box(low=0.0, high=10.0, shape=())

    def reset(self, *, key: PRNGKeyArray) -> State:
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

    def step(self, state: State, action: Array) -> State:
        count = state.env_state + 1.0
        return replace(
            state,
            env_state=count,
            observation=count,
            step_count=state.step_count + 1,
        )


class _TeamEnv:
    """Fully annotated three-agent env."""

    num_agents = 3

    @property
    def action_space(self) -> Space:
        return stack_space(Discrete(2), self.num_agents)

    @property
    def observation_space(self) -> Space:
        return stack_space(Box(low=0.0, high=10.0, shape=()), self.num_agents)

    def reset(self, *, key: PRNGKeyArray) -> MARLState:
        return MARLState(
            env_state=jnp.float32(0.0),
            agents=Agents(
                observation=jnp.zeros(self.num_agents),
                active=jnp.ones(self.num_agents, dtype=bool),
                reward=jnp.zeros(self.num_agents),
            ),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state: MARLState, action: Array) -> MARLState:
        return replace(
            state,
            agents=replace(state.agents, reward=action.astype(jnp.float32)),
            step_count=state.step_count + 1,
        )


@jax.tree_util.register_dataclass
@dataclass
class _MaskedState(State):
    """Custom user state with an extra field, the pattern shown in the README."""

    action_mask: Array


class _MaskedEnv:
    @property
    def action_space(self) -> Space:
        return Discrete(2)

    @property
    def observation_space(self) -> Space:
        return Box(low=0.0, high=10.0, shape=())

    def reset(self, *, key: PRNGKeyArray) -> _MaskedState:
        return _MaskedState(
            env_state=jnp.float32(0.0),
            observation=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
            action_mask=jnp.ones(2, dtype=bool),
        )

    def step(self, state: _MaskedState, action: Array) -> _MaskedState:
        return replace(state, step_count=state.step_count + 1)


def test_sarl_env_satisfies_env() -> None:
    env: Env = _CounterEnv()
    state = env.reset(key=jax.random.key(0))
    assert_type(state, State)
    assert isinstance(state, State)


def test_marl_env_satisfies_marl_env() -> None:
    env: MARLEnv = _TeamEnv()
    state = env.reset(key=jax.random.key(0))
    assert_type(state, MARLState)
    assert_type(env.num_agents, int)
    assert env.num_agents == 3


def test_vmap_sarl_env_satisfies_vector_env() -> None:
    env: VectorEnv = VmapWrapper(_CounterEnv(), num_envs=4)
    assert_type(env.num_envs, int)
    assert env.num_envs == 4


def test_vmap_marl_env_satisfies_marl_vector_env() -> None:
    wrapper = VmapWrapper(_TeamEnv(), num_envs=4)
    assert_type(wrapper, VmapWrapper[MARLState])
    env: MARLVectorEnv = wrapper
    assert env.num_agents == 3
    assert env.num_envs == 4


def test_wrapper_num_agents_is_int() -> None:
    wrapper = TimeLimit(_CounterEnv(), max_steps=10)
    assert_type(wrapper.num_agents, int)
    assert wrapper.num_agents == 1


def test_custom_state_env_satisfies_env() -> None:
    env: Env[_MaskedState] = _MaskedEnv()
    state = env.reset(key=jax.random.key(0))
    assert_type(state, _MaskedState)
    assert state.action_mask.shape == (2,)


def test_custom_state_survives_wrapping() -> None:
    env = VmapWrapper(TimeLimit(_MaskedEnv(), max_steps=8), num_envs=4)
    assert_type(env, VmapWrapper[_MaskedState])
    state = env.reset(key=jax.random.key(0))
    assert_type(state, _MaskedState)
    assert state.action_mask.shape == (4, 2)


def test_selective_reset_round_trips_marl_state() -> None:
    env = VmapWrapper(_TeamEnv(), num_envs=4)
    state = env.reset(key=jax.random.key(0))
    state = env.step(state, jnp.ones((4, 3), dtype=jnp.int32))
    state = env.reset(key=jax.random.key(1), state=state, done=state.done)
    assert_type(state, MARLState)
    assert state.agents.reward.shape == (4, 3)
