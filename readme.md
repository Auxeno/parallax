<div align="center">

  <h1> Parallax </h1>

  <h3>A JAX Reinforcement Learning Protocol</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Parallax is a minimal JAX reinforcement learning API for SARL and MARL environments. It defines a protocol - not a framework - so you can adapt it to your problem rather than adapting your problem to it.

- **Protocol-based**: satisfy the interface, no inheritance required
- **JAX-native**: pure functions, pytree states, `jit`/`vmap` just work
- **SARL and MARL**: same API, agents are just a batch dimension
- **Flexible observations**: flat arrays, dicts, named tuples, graphs with masks, anything that's a pytree

## Install

```bash
pip install parallax-api
```

## API

### Protocol

```python
class Env(Protocol):
    action_space: Space
    observation_space: Space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]: ...
    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]: ...
```

### Core Types

```python
@dataclass
class EnvState:
    state: PyTree          # your environment's internal state
    step: Int[Array, ""]   # current timestep count
    key: PRNGKeyArray | None  # RNG key (None if env is deterministic)

@dataclass
class Timestep:
    observation: Float[Array, "..."] | PyTree  # what the agent sees
    reward: Float[Array, "..."]     # reward signal
    termination: Bool[Array, "..."] # natural episode end
    truncation: Bool[Array, "..."]  # forced episode end (e.g. time limit)
    info: PyTree                    # auxiliary data, logs, metrics
```

Both are registered as JAX pytrees.

### Spaces

| Space | Description |
|-------|-------------|
| `Discrete(n, shape=())` | Integer(s) in `[0, n)`, pass `shape` for batched sampling |
| `Box(low, high, shape)` | Continuous values in `[low, high]` |
| `MultiDiscrete(actions_per_dim)` | Vector of independent discrete values (shape derived from `actions_per_dim`) |
| `MultiBinary(n)` | Binary vector of length `n` |
| `PyTreeSpace(spaces)` | Pytree of spaces - for structured observations |

All spaces implement `sample(key)` for random sampling.

## Example

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from parallax import Env, EnvState, Timestep, Discrete, Box

@jax.tree_util.register_dataclass
@dataclass
class GridState:
    pos: jnp.ndarray  # (2,) agent position

class GridWorld:
    action_space = Discrete(4)         # up/down/left/right
    observation_space = Box(0.0, 1.0, (5, 5))

    def reset(self, *, key):
        pos = jax.random.randint(key, (2,), 0, 5)
        state = EnvState(
            state=GridState(pos=pos),
            step=jnp.array(0),
            key=key,
        )
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

## MARL

Same protocol. Agents are a dimension on your arrays. Reward, termination, and truncation have shape `(num_agents,)`:

```python
import jax.numpy as jnp
from parallax import MultiDiscrete, Box

num_agents = 3

class MultiAgentEnv:
    action_space = MultiDiscrete(jnp.full(num_agents, 5))
    observation_space = Box(0.0, 1.0, (num_agents, 4))

    def reset(self, *, key):
        # observation: (num_agents, 4)
        # reward: (num_agents,), termination: (num_agents,), truncation: (num_agents,)
        ...

    def step(self, state, action):
        # action: (num_agents,)
        ...
```

Spaces reflect the full shape including the agent dimension. `jax.vmap` over the env axis for batched environments, over the agent axis for per-agent logic - compose as needed.

## Structured Observations

For graph-structured or entity-based observations, use `PyTreeSpace`:

```python
from parallax import PyTreeSpace, Box, MultiBinary

observation_space = PyTreeSpace({
    "nodes": Box(0.0, 1.0, (num_nodes, node_features)),
    "edges": Box(0.0, 1.0, (num_edges, edge_features)),
    "action_mask": MultiBinary(num_actions),
    "edge_mask": MultiBinary(num_edges),
})

# sample() returns a dict with the same structure
obs = observation_space.sample(key=key)
```

## Wrappers

### VmapWrapper

Vectorises an environment over a batch dimension. Pass `done` to selectively reset only finished environments.

```python
from parallax import VmapWrapper

env = VmapWrapper(GridWorld())

# full reset: pass (num_envs, 2) keys
keys = jax.random.split(jax.random.key(0), 128)
state, timestep = env.reset(key=keys)

# step: batched
state, timestep = env.step(state, actions)

# reset only done envs (after you've handled bootstrapping)
done = timestep.termination | timestep.truncation
state, timestep = env.reset(key=jax.random.key(1), state=state, timestep=timestep, done=done)
```

### TimeLimit

Sets truncation after a fixed number of steps. Relies on the env incrementing `state.step` each step.

```python
from parallax import TimeLimit

env = TimeLimit(GridWorld(), max_steps=200)
```

Compose them:

```python
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200))
```

## Training Loop

```python
env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200))
num_envs = 128

@jax.jit
def train_step(state, timestep, key):
    action_key, reset_key = jax.random.split(key)

    # your policy here
    actions = jax.vmap(env.action_space.sample)(key=jax.random.split(action_key, num_envs))

    # step
    state, timestep = env.step(state, actions)

    # bootstrap value estimates from the *current* observation before resetting,
    # since reset will overwrite observations for done envs
    done = timestep.termination | timestep.truncation
    # v_bootstrap = jax.vmap(value_fn)(timestep.observation)

    # reset done envs
    state, timestep = env.reset(key=reset_key, state=state, timestep=timestep, done=done)
    return state, timestep
```

## Scope

Parallax assumes fixed-shape tensors, simultaneous actions, and a pure functional step. This fits:

- Single-agent RL
- Simultaneous-move MARL (PettingZoo `parallel_env` style)
- Turn-based games (chess, Go), one `step` per move, opponent is part of the environment
- Differentiable environments (`jax.grad` through `step`)
- Model-based RL, a learned dynamics model has the same signature
- Structured and masked observations via pytrees

It does not naturally fit:

- **AEC**, where multiple agents act sequentially within a round and observe each other's intermediate actions (e.g. poker, auctions, negotiation games)
- **Variable agent counts**, JAX requires fixed shapes so agents entering or leaving requires padding and masking
