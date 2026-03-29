<div align="center">

  <h1> Parallax </h1>

  <h3>A JAX Reinforcement Learning Protocol</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Why Parallax?

JAX RL environments need pure functions and immutable state, but there's no standard for what that looks like. Parallax defines a minimal `reset`/`step` contract so any environment exposes the same interface.

- **For JAX RL users**: Write agents, experience collection, and training loops once. Swap environments without changing your code.
- **For Gymnasium users**: The same familiar concepts (reset, step, observation, reward) rebuilt for JAX. Pure functions instead of mutable objects, so everything works with `jit`, `vmap`, and `scan`.

Protocol, not a framework. No base class, no registration.

## Install

```bash
pip install parallax-rl
```

## Quick Start

```python
import jax

env = GridWorld()
state = env.reset(key=jax.random.key(0))

for _ in range(200):
    action = agent(state.observation)
    state = env.step(state, action)

    if state.done:
        break
```

## How It Works

RL environments are conventionally stateful (Gymnasium, PettingZoo, etc.). Calling `env.step()` mutates the environment in place. JAX needs pure functions and immutable data, so Parallax splits things in two:

**Env** is stateless. It has two pure functions (`reset` and `step`) with no internal state.

**State** is a JAX pytree that holds all the data. Every call to `reset` or `step` returns a new State with precomputed fields:

```python
state = env.reset(key=jax.random.key(0))
state = env.step(state, action)

state.env_state    # raw environment data (any pytree)
state.observation  # what the agent sees
state.reward       # scalar reward
state.termination  # episode ended naturally
state.truncation   # episode was cut short
state.done         # termination | truncation
state.info         # extra metadata (dict)
state.step_count   # current timestep
state.key          # JAX RNG key
```

State is pure data. All values are computed in `reset`/`step` and stored directly.

## Building an Environment

Implement `reset` and `step`. Each returns a `State` with all fields computed:

```python
import jax
import jax.numpy as jnp
from typing import NamedTuple
from jaxtyping import Array, PRNGKeyArray
from parallax import Space, State, spaces


class GridState(NamedTuple):
    pos: Array
    goal: Array


class GridWorld:
    action_space: Space = spaces.Discrete(4)
    observation_space: Space = spaces.Box(0.0, 4.0, (4,))

    def reset(self, *, key: PRNGKeyArray) -> State:
        key, goal_key = jax.random.split(key)
        pos = jnp.zeros(2, dtype=jnp.float32)
        goal = jax.random.randint(goal_key, (2,), minval=1, maxval=5).astype(jnp.float32)
        return State(
            env_state=GridState(pos=pos, goal=goal),
            observation=jnp.concatenate([pos, goal]),
            reward=jnp.float32(0.0),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
        )

    def step(self, state: State, action: Array) -> State:
        moves = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.float32)
        pos = jnp.clip(state.env_state.pos + moves[action], 0.0, 4.0)
        goal = state.env_state.goal
        return State(
            env_state=GridState(pos=pos, goal=goal),
            observation=jnp.concatenate([pos, goal]),
            reward=jnp.exp(-jnp.linalg.norm(pos - goal)),
            termination=jnp.all(pos == goal),
            truncation=jnp.bool_(False),
            info={},
            step_count=state.step_count + 1,
            key=state.key,
        )
```

`env_state` is your raw environment data and can be any JAX pytree. The other fields (`observation`, `reward`, etc.) are derived from it in `reset`/`step`.

For multi-agent environments, agents are a dimension on your arrays. Reward, termination, and truncation become shape `(num_agents,)` while the method signatures stay the same. Environments where agents have different action space sizes will need padding and masking to maintain fixed array shapes. This is a JAX constraint (need for fixed shapes) rather than a Parallax one.

## Wrappers

Wrappers compose to add functionality:

```python
from parallax import AutoResetWrapper, TimeLimit, VmapWrapper

num_envs = 128
env = VmapWrapper(AutoResetWrapper(TimeLimit(GridWorld(), max_steps=200)), num_envs=num_envs)
state = env.reset(key=jax.random.key(0))
state = env.step(state, actions)
```

For manual resets (e.g. when you need terminal observations for value bootstrapping):

```python
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200), num_envs=num_envs)
state = env.step(state, actions)
state = env.reset(key=reset_key, state=state, done=state.done)
```

## Adapters

Use existing JAX RL environments with Parallax via adapters:

```python
import gymnax
from parallax.adapters import GymnaxAdapter

env = GymnaxAdapter(gymnax.make("CartPole-v1")[0])
env = VmapWrapper(env, num_envs=128)
```

```python
import brax.envs
from parallax.adapters import BraxAdapter

env = BraxAdapter(brax.envs.get_environment("ant"))
env = VmapWrapper(env, num_envs=128)
```

Adapters map foreign reset/step APIs to the Parallax protocol. Brax's built-in auto-reset is stripped automatically to preserve terminal observations.

## Custom Properties

Subclass `State` to add extra fields. For example, adding an action mask to `GridWorld`:

```python
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass
class MaskedState(State):
    action_mask: Bool[Array, "4"]
```

Then return `MaskedState` from your env's `reset` and `step`:

```python
class MaskedGridWorld(GridWorld):
    def reset(self, *, key: PRNGKeyArray) -> MaskedState:
        state = super().reset(key=key)
        return MaskedState(**vars(state), action_mask=compute_mask(state.env_state))

    def step(self, state: MaskedState, action: Array) -> MaskedState:
        state = super().step(state, action)
        return MaskedState(**vars(state), action_mask=compute_mask(state.env_state))

state.action_mask  # fully typed, works with jit/vmap/wrappers
```

## Collecting Experience

Use `jax.lax.scan` for vectorized rollouts. Manual resets let you capture terminal observations before resetting done environments, which is needed for value bootstrapping:

```python
from dataclasses import dataclass
from parallax import VmapWrapper

@jax.tree_util.register_dataclass
@dataclass
class Experience:
    observation: jax.Array
    next_observation: jax.Array
    action: jax.Array
    reward: jax.Array
    termination: jax.Array
    
num_envs = 128
env = VmapWrapper(GridWorld(), num_envs=num_envs)

key = jax.random.key(0)
key, reset_key = jax.random.split(key)
state = env.reset(key=reset_key)
obs = state.observation

def step_fn(carry, _):
    state, obs, key = carry
    key, action_key, reset_key = jax.random.split(key, 3)
    action = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, num_envs))

    state = env.step(state, action)
    next_obs = state.observation

    experience = Experience(
        observation=obs,
        next_observation=next_obs,
        action=action,
        reward=state.reward,
        termination=state.termination,
    )

    # Reset environments where done, terminal obs captured above
    state = env.reset(key=reset_key, state=state, done=state.done)
    obs = state.observation

    return (state, obs, key), experience

(state, obs, key), experiences = jax.lax.scan(step_fn, (state, obs, key), None, length=256)
```
