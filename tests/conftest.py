import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

from parallax.spaces import Box, Discrete, MultiBinary, MultiDiscrete, PyTreeSpace
from parallax.core import State


@pytest.fixture
def key():
    return jax.random.key(0)


def _is_prng_key(x):
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)


def assert_trees_close(tree1, tree2, rtol=1e-5, atol=1e-5):
    for a, b in zip(jtu.tree_leaves(tree1), jtu.tree_leaves(tree2)):
        if _is_prng_key(a):
            npt.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            npt.assert_allclose(a, b, rtol=rtol, atol=atol)


def assert_trees_equal(tree1, tree2):
    for a, b in zip(jtu.tree_leaves(tree1), jtu.tree_leaves(tree2)):
        if _is_prng_key(a):
            npt.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        else:
            npt.assert_array_equal(a, b)


class CountingEnv:
    """Minimal env: state is a scalar counter, obs = count, reward = 1 each step.
    Terminates when count >= 10."""

    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=100.0, shape=())

    def observation(self, state):
        return state.env_state.astype(jnp.float32)

    def reward(self, state):
        return jnp.where(state.step_count > 0, jnp.float32(1.0), jnp.float32(0.0))

    def termination(self, state):
        return state.env_state >= 10.0

    def truncation(self, state):
        return jnp.bool_(False)

    def info(self, state):
        return {}

    def reset(self, *, key):
        return State(self, env_state=jnp.float32(0.0), step_count=jnp.int32(0), key=key)

    def step(self, state, action):
        new_count = state.env_state + 1.0
        return State(self, env_state=new_count, step_count=state.step_count + 1, key=state.key)


class GridWorldEnv:
    """2D grid env: state is (x, y) position, actions move in 4 cardinal directions.
    Terminates when agent reaches (4, 4)."""

    action_space = Discrete(4)
    observation_space = Box(low=0.0, high=4.0, shape=(2,))

    def observation(self, state):
        return state.env_state

    def reward(self, state):
        at_goal = jnp.all(state.env_state == jnp.array([4.0, 4.0]))
        return jnp.where(at_goal, 1.0, 0.0)

    def termination(self, state):
        return jnp.all(state.env_state == jnp.array([4.0, 4.0]))

    def truncation(self, state):
        return jnp.bool_(False)

    def info(self, state):
        return {}

    def reset(self, *, key):
        pos = jnp.zeros(2, dtype=jnp.float32)
        return State(self, env_state=pos, step_count=jnp.int32(0), key=key)

    def step(self, state, action):
        moves = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.float32)
        new_pos = jnp.clip(state.env_state + moves[action], 0.0, 4.0)
        return State(self, env_state=new_pos, step_count=state.step_count + 1, key=state.key)


_ENV_NAMES = ["counting", "grid_world"]


def _build_envs():
    return {"counting": CountingEnv(), "grid_world": GridWorldEnv()}


@pytest.fixture(params=_ENV_NAMES)
def env(request):
    return _build_envs()[request.param]


_SPACE_CONFIGS = {
    "discrete": Discrete(5),
    "discrete_batched": Discrete(3, shape=(4,)),
    "box": Box(low=-1.0, high=1.0, shape=(8,)),
    "box_scalar": Box(low=0.0, high=1.0, shape=()),
    "multi_discrete": MultiDiscrete(actions_per_dim=jnp.array([3, 5, 2])),
    "multi_binary": MultiBinary(n=6),
    "pytree": PyTreeSpace(
        spaces={"pos": Box(low=-1.0, high=1.0, shape=(2,)), "action": Discrete(4)}
    ),
}

_SPACE_NAMES = list(_SPACE_CONFIGS.keys())


@pytest.fixture(params=_SPACE_NAMES)
def space(request):
    return _SPACE_CONFIGS[request.param]
