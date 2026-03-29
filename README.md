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

Traditional RL environments are stateful. Calling `env.step()` mutates the environment in place. JAX needs pure functions and immutable data, so Parallax splits things in two:

**Env** is stateless. A collection of pure functions (`reset`, `step`, `observation`, `reward`, ...) with no internal state.

**State** is a JAX pytree that holds all the data. It gets passed into and returned from env functions.

```python
state = env.reset(key=jax.random.key(0))
state = env.step(state, action)

state.observation  # what the agent sees
state.reward       # scalar reward
state.termination  # episode ended naturally
state.truncation   # episode was cut short
state.done         # termination | truncation
state.info         # extra metadata
```

Properties are computed lazily. Accessing `state.reward` calls `env.reward(state)` under the hood. If you never read a property, the computation doesn't happen.

## Building an Environment

Implement `reset` and `step` for dynamics, plus property methods for what agents observe:

```python
import jax
import jax.numpy as jnp
from typing import NamedTuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray
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
        return State(self, env_state=GridState(pos=pos, goal=goal), step_count=jnp.int32(0), key=key)

    def step(self, state: State, action: Array) -> State:
        moves = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.float32)
        pos = jnp.clip(state.env_state.pos + moves[action], 0.0, 4.0)
        env_state = state.env_state._replace(pos=pos)
        return State(self, env_state=env_state, step_count=state.step_count + 1, key=state.key)

    def observation(self, state: State) -> Float[Array, "4"]:
        return jnp.concatenate([state.env_state.pos, state.env_state.goal])

    def reward(self, state: State) -> Float[Array, ""]:
        return jnp.exp(-jnp.linalg.norm(state.env_state.pos - state.env_state.goal))

    def termination(self, state: State) -> Bool[Array, ""]:
        return jnp.all(state.env_state.pos == state.env_state.goal)

    def truncation(self, state: State) -> Bool[Array, ""]:
        return jnp.bool_(False)

    def info(self, state: State) -> dict:
        return {}
```

`env_state` is your raw environment data and can be any JAX pytree. Property methods like `observation` and `reward` derive what the agent sees from that internal state.

For multi-agent environments, agents are a dimension on your arrays. Reward, termination, and truncation become shape `(num_agents,)` while the method signatures stay the same. Environments where agents have different action space sizes will need padding and masking to maintain fixed array shapes. This is a JAX constraint (need for fixed shapes) rather than a Parallax one.

## Wrappers

Wrappers compose to add functionality:

```python
from parallax import AutoResetWrapper, TimeLimit, VmapWrapper

num_envs = 128
env = VmapWrapper(AutoResetWrapper(TimeLimit(GridWorld(), max_steps=200)))
keys = jax.random.split(jax.random.key(0), num_envs)
state = env.reset(key=keys)
state = env.step(state, actions)
```

For manual resets (e.g. when you need terminal observations for value bootstrapping):

```python
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200))
state = env.step(state, actions)
state = env.reset(key=reset_key, state=state, done=state.done)
```

## Custom Properties

Any method on your env is accessible as a property on the state. This works the same way as the built-in properties like `observation` and `reward`:

```python
class MaskedGridWorld(GridWorld):
    def action_mask(self, state: State) -> Bool[Array, "4"]:
        pos = state.env_state.pos
        return jnp.array([pos[1] < 4, pos[1] > 0, pos[0] < 4, pos[0] > 0])

state.action_mask  # calls env.action_mask(state)
```

Custom properties forward through wrappers automatically, including `VmapWrapper`.

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
env = VmapWrapper(GridWorld())

key = jax.random.key(0)
key, reset_key = jax.random.split(key)
state = env.reset(key=jax.random.split(reset_key, num_envs))
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
    state = env.reset(key=jax.random.split(reset_key, num_envs), state=state, done=state.done)
    obs = state.observation

    return (state, obs, key), experience

(state, obs, key), experiences = jax.lax.scan(step_fn, (state, obs, key), None, length=256)
```
