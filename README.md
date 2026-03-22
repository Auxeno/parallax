<div align="center">

  <h1> Parallax </h1>

  <h3>A JAX Reinforcement Learning Protocol</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Parallax is a protocol for JAX RL environments. It defines a shared `reset`/`step` contract, not a framework or base class, so any environment can expose the same interface while carrying whatever data its agents actually need.

## Why

Every JAX RL library defines its own environment interface. Gymnax, Jumanji, Brax, PureJaxRL all have their own state types, timestep types, and step signatures. Training code written for one doesn't transfer to another without adapter layers.

The underlying loop is always the same: reset, observe, act, step, repeat. The variation is in *what data the environment provides*, not in *how the loop works*.

Parallax fixes the contract and leaves the data flexible. The base `Timestep` carries the fields every RL loop needs. When an environment needs more, it extends `Timestep` with new fields. The `reset`/`step` signatures stay the same.

- **Protocol, not a framework.** Satisfy the interface, no inheritance required.
- **JAX-native.** Pure functions, pytree states, `jit`/`vmap`/`grad` just work.
- **SARL and MARL.** Same interface, agents are a batch dimension.
- **Extensible timesteps.** Subclass `Timestep` to carry any additional data.
- **Gymnasium-style spaces.** `Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`, `PyTreeSpace`.

## Install

```bash
pip install parallax-rl
```

```python
import parallax
```

## The Protocol

```python
class Env(Protocol):
    action_space: Space
    observation_space: Space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]: ...
    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]: ...
```

Any class with these methods and properties satisfies the protocol. No base class, no registration.

```python
@dataclass
class EnvState:
    state: PyTree          # your environment's internal state
    step: Int[Array, ""]   # current timestep count
    key: PRNGKeyArray      # RNG key

@dataclass
class Timestep:
    observation: Float[Array, "..."] | PyTree
    reward: Float[Array, "..."]
    termination: Bool[Array, "..."]
    truncation: Bool[Array, "..."]
    info: PyTree
```

Both are registered as JAX pytrees.

## Example

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from parallax import EnvState, Timestep, spaces

@jax.tree_util.register_dataclass
@dataclass
class GridState:
    pos: jnp.ndarray

class GridWorld:
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(0.0, 1.0, (5, 5))

    def reset(self, *, key):
        pos = jax.random.randint(key, (2,), 0, 5)
        state = EnvState(state=GridState(pos=pos), step=jnp.array(0), key=key)
        obs = jnp.zeros((5, 5)).at[pos[0], pos[1]].set(1.0)
        return state, Timestep(obs, jnp.float32(0), jnp.False_, jnp.False_, {})

    def step(self, state, action):
        moves = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        new_pos = jnp.clip(state.state.pos + moves[action], 0, 4)
        at_goal = jnp.all(new_pos == jnp.array([4, 4]))
        new_state = EnvState(
            state=GridState(pos=new_pos),
            step=state.step + 1,
            key=state.key,
        )
        obs = jnp.zeros((5, 5)).at[new_pos[0], new_pos[1]].set(1.0)
        return new_state, Timestep(obs, at_goal.astype(jnp.float32), at_goal, jnp.False_, {})
```

### Extending Timestep

When your environment needs to provide more data, subclass `Timestep`:

```python
@jax.tree_util.register_dataclass
@dataclass
class MaskedTimestep(Timestep):
    action_mask: Bool[Array, "num_actions"]
```

Your agent accesses `timestep.action_mask` directly. The same pattern works for visibility masks, communication channels, global state for a centralised critic, or anything else.

### Multi-Agent

For MARL, agents are a dimension on your arrays. Reward, termination, and truncation have shape `(num_agents,)` for per-agent signals, or remain scalar for shared team rewards. The `step` and `reset` signatures don't change.

## Wrappers

```python
from parallax import AutoResetWrapper, TimeLimit, VmapWrapper

# Auto-reset on done, with a step limit, across 128 parallel envs
env = VmapWrapper(AutoResetWrapper(TimeLimit(GridWorld(), max_steps=200)))
keys = jax.random.split(jax.random.key(0), 128)
state, timestep = env.reset(key=keys)
state, timestep = env.step(state, actions)

# Manual reset when you need terminal observations (e.g. value bootstrapping)
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200))
state, timestep = env.step(state, actions)
done = timestep.termination | timestep.truncation
state, timestep = env.reset(key=reset_key, state=state, timestep=timestep, done=done)
```

## Scope

Parallax assumes fixed-shape tensors, simultaneous actions, and a pure functional step. This covers single-agent RL, simultaneous-move MARL, turn-based games, and differentiable environments.

It is not a good fit for AEC (sequential agent actions within a round) or variable-dimension action spaces across steps (masking is fine, changing the shape requires padding).
