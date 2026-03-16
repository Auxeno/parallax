<div align="center">

  <h1> Parallax </h1>

  <h3>A JAX Reinforcement Learning Protocol</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Parallax is a minimal reinforcement learning protocol for JAX. It defines a shared contract for environment `reset` and `step`, not a framework or base class, so any environment can expose the same interface while packaging whatever data its agents actually need.

## The Problem

Every JAX RL library defines its own environment interface. Gymnax, Jumanji, Brax, and PureJaxRL all have their own state types, timestep types, and step signatures. Training code written for one doesn't transfer to another without adapter layers.

MARL makes this worse. Different problems need action masks, visibility masks, team rewards, per-agent rewards, or structured observations, but the underlying loop is always the same: reset, observe, act, step, repeat. The variation is in *what data the environment provides*, not in *how the loop works*.

## How Parallax Solves This

Parallax keeps the `reset`/`step` contract fixed and minimal. The base `Timestep` carries the fields every RL loop needs. When an environment needs more, it extends `Timestep` with new fields. An environment with action masking adds an `action_mask` field; a partially observable environment adds a `visibility` field. Agents access these directly as `timestep.action_mask` or `timestep.visibility`.

- **Protocol, not a framework**: satisfy the interface, no inheritance required
- **JAX-native**: pure functions, pytree states, `jit`/`vmap`/`grad` just work
- **SARL and MARL**: same interface, agents are a batch dimension
- **Extensible timesteps**: add fields as your environment needs them

## Install

```bash
pip install parallax-rl
```

## The Protocol

### Env

```python
class Env(Protocol):
    action_space: Space
    observation_space: Space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]: ...
    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]: ...
```

Any class with these methods and properties satisfies the protocol. No base class, no registration.

### EnvState

```python
@dataclass
class EnvState:
    state: PyTree
    step: Int[Array, ""]
    key: PRNGKeyArray
```

`state` holds your environment's internal state as any pytree. `step` tracks the current timestep count. `key` carries the RNG key.

### Timestep

```python
@dataclass
class Timestep:
    observation: Float[Array, "..."] | PyTree
    reward: Float[Array, "..."]
    termination: Bool[Array, "..."]
    truncation: Bool[Array, "..."]
    info: PyTree
```

Both `EnvState` and `Timestep` are registered as JAX pytrees. When your environment needs to provide more data, extend `Timestep`:

```python
@jax.tree_util.register_dataclass
@dataclass
class MaskedTimestep(Timestep):
    action_mask: Bool[Array, "..."]
```

Your agent accesses `timestep.action_mask` directly. The `reset`/`step` signatures stay the same.

### Spaces

| Space | Description |
|-------|-------------|
| `Discrete(n, shape=())` | Integer(s) in `[0, n)`, pass `shape` for batched sampling |
| `Box(low, high, shape)` | Continuous values in `[low, high]` |
| `MultiDiscrete(actions_per_dim)` | Vector of independent discrete values |
| `MultiBinary(n)` | Binary vector of length `n` |
| `PyTreeSpace(spaces)` | Pytree of spaces, for structured observations |

All spaces implement `sample(*, key)` for random sampling.

## Examples

### Single-Agent

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from parallax import EnvState, Timestep, Discrete, Box

@jax.tree_util.register_dataclass
@dataclass
class GridState:
    pos: jnp.ndarray

class GridWorld:
    action_space = Discrete(4)
    observation_space = Box(0.0, 1.0, (5, 5))

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

### Multi-Agent

For MARL, agents are a dimension on your arrays. Reward, termination, and truncation have shape `(num_agents,)` for per-agent signals, or remain scalar for shared team rewards. The `step` and `reset` signatures don't change.

### Action Masking

Extend `Timestep` with an `action_mask` field. The agent accesses it directly.

```python
@jax.tree_util.register_dataclass
@dataclass
class MaskedTimestep(Timestep):
    action_mask: Bool[Array, "num_actions"]

class MaskedActionEnv:
    action_space = Discrete(9)
    observation_space = Box(0.0, 1.0, (3, 3))

    def step(self, state, action):
        ...
        return new_state, MaskedTimestep(
            observation=board_state,
            reward=jnp.float32(reward),
            termination=jnp.bool_(done),
            truncation=jnp.False_,
            info={},
            action_mask=legal_moves,
        )
```

```python
logits = policy_network(timestep.observation)
masked_logits = jnp.where(timestep.action_mask, logits, -1e9)
action = jax.random.categorical(key, masked_logits)
```

The same pattern works for visibility masks, communication channels, global state for a centralised critic, or anything else. Extend `Timestep`, access the field.

## Wrappers

### AutoResetWrapper

Automatically resets done environments after each step. Use this when you don't need to handle truncation separately or bootstrap from terminal observations.

```python
from parallax import AutoResetWrapper, TimeLimit, VmapWrapper

env = VmapWrapper(AutoResetWrapper(TimeLimit(GridWorld(), max_steps=200)))
keys = jax.random.split(jax.random.key(0), 128)
state, timestep = env.reset(key=keys)
state, timestep = env.step(state, actions)
```

If you need the terminal observation before it gets overwritten (e.g. for value bootstrapping), use `VmapWrapper` directly and reset manually:

```python
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200))

state, timestep = env.step(state, actions)
done = timestep.termination | timestep.truncation
state, timestep = env.reset(key=reset_key, state=state, timestep=timestep, done=done)
```

### TimeLimit

Sets truncation after a fixed number of steps.

```python
from parallax import TimeLimit

env = TimeLimit(GridWorld(), max_steps=200)
```

## Scope

Parallax assumes fixed-shape tensors, simultaneous actions, and a pure functional step. This covers:

- Single-agent RL
- Simultaneous-move MARL (PettingZoo `parallel_env` style)
- Turn-based games where one `step` is one move and the opponent is part of the environment
- Differentiable environments and model-based RL

It does not naturally fit:

- **AEC**, where agents act sequentially within a round and observe each other's intermediate actions
- **Variable action space sizes across steps** (changing which actions are legal is fine via masking, but changing the dimension itself requires padding)
