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

# With adapter dependencies
pip install parallax-rl[brax]     # Brax environments
pip install parallax-rl[gymnax]   # Gymnax environments
pip install parallax-rl[mjx]      # MuJoCo Playground (MJX) environments
pip install parallax-rl[adapters] # All adapters
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

For multi-agent environments, Parallax has a dedicated protocol. See [Multi-Agent Environments](#multi-agent-environments).

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

## Multi-Agent Environments

Multi-agent APIs typically conflate two different things: the episode ending and an individual agent dying. Parallax keeps them separate. `MARLState` keeps termination and truncation as episode-level scalars and tracks per-agent lifecycle with an `active` mask:

```python
from parallax import MARLEnv, MARLState, Agents

state.env_state           # raw environment data (any pytree)
state.termination         # the MDP ended naturally
state.truncation          # the MDP was cut short
state.done                # termination | truncation

state.agents.observation  # per-agent observations, leading dim num_agents
state.agents.active       # which agents act on the next step
state.agents.reward       # per-agent rewards, zeros in inactive slots
state.agents.action_mask  # per-agent legal actions, or None
state.global_observation  # full-state view for centralised critics, or None
```

Agents occupy array slots `0..num_agents - 1` and are shape-homogeneous: `action_space` and `observation_space` describe a single agent's space, and per-agent data stacks on a leading `num_agents` dimension. Heterogeneous agents pad to the largest shape and express legality through `action_mask`.

The contract environments implement:

1. Inactive agents' actions are ignored, environments tolerate arbitrary values in those slots
2. Inactive agents' observations are zeros
3. Inactive agents' slots in `agents.reward` are zeros, the step an agent dies on is an active step and may carry reward
4. Inactive agents' action masks contain at least one legal action
5. `agents.active` marks the agents whose actions the next `step` call will use
6. The environment sets `termination` itself, all agents becoming inactive does not implicitly end the MDP

Rule 5 fixes the timing: rewards in a state pair with the `active` mask of the previous state.

```python
state_1 = env.step(state_0, actions_0)
# state_0.agents.active  marks whose actions in actions_0 are used
# state_1.agents.reward  pairs with state_0.agents.active
```

Everything a learning algorithm needs is a one-liner from these fields, nothing is stored twice:

```python
valid = state_t.agents.active                            # transition is real, use it in the loss
bootstrap = state_t1.agents.active & ~state_t1.termination  # died or terminated, no bootstrap
rewards = state_t1.agents.reward                         # pairs with state_t.agents.active
```

An agent that dies on the step the episode truncates composes correctly with no special casing: it stops bootstrapping through `active`, while surviving agents bootstrap from the terminal observation.

Teams are not a protocol concept, slots are. Competitive and team tasks broadcast each team's reward to its members' slots in `agents.reward`: all slots equal is fully cooperative, `+1`/`-1` across two blocks of slots is two-team zero-sum. A team critic reads its own members' slots, and team membership itself is environment metadata (subclass `Agents` to expose it, see [Custom Properties](#custom-properties)). There is no MDP-level reward field, the shared scalar in cooperative tasks is simply the value at any active slot.

Because the episode-level fields match the single-agent `State`, wrappers like `TimeLimit` and `VmapWrapper` work on multi-agent environments unchanged.

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

```python
from mujoco_playground import registry
from parallax.adapters import MJXAdapter

env = MJXAdapter(registry.load("HumanoidWalk", config_overrides={"impl": "jax"}))
env = VmapWrapper(env, num_envs=128)
```

Adapters map foreign reset/step APIs to the Parallax protocol. Brax and MJX adapters extract episode length from the underlying environment and handle truncation internally. Brax's built-in auto-reset is stripped automatically to preserve terminal observations.

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

## Assumptions & Sharp Edges

Parallax trusts environments to follow the protocol. Nothing is validated at runtime, and the jaxtyping annotations are documentation rather than enforcement. The things most likely to bite:

**General**

- **Pytree structure must be stable.** Every `reset` and `step` of a given environment must return the same treedef: same `info` keys, same dtypes, and optional fields consistently `None` or consistently arrays. Structure that changes between calls breaks `jit`, `lax.cond`, `lax.scan`, and selective resets.
- **Shapes are fixed.** Anything variable-length (legal actions, live agents, episode length) is expressed with padding and masking, never with dynamic shapes. This is a JAX constraint, not a Parallax choice.
- **`step_count` powers `TimeLimit`.** Environments must increment it every step or wrapped truncation never fires.
- **Termination and truncation are distinct on purpose.** Bootstrapping value targets from the terminal observation is correct on truncation and wrong on termination. `done` alone is not enough to write a correct update, it only tells you the episode ended.
- **Stepping a done state is undefined by the protocol.** Environments are not required to freeze or reset themselves. Reset instead, selective reset via `VmapWrapper` keeps terminal observations available first.
- **Shapes widen under `VmapWrapper`.** A scalar `reward` becomes `(num_envs,)` and multi-agent `(num_agents,)` fields become `(num_envs, num_agents)`. The `...` in the field annotations is that batch dimension.

**Multi-agent**

- **The agent count is static.** `num_agents` is a trace-time constant and an upper bound. Agents can die, and the protocol does not forbid re-activation (respawns), but the slot count never changes.
- **The reward/active pairing is off by one by design.** Rewards in the state returned by `step` pair with the `active` mask of the state passed in. Scan-collected trajectories must respect this, see the timing diagram above.
- **There is no MDP-level reward.** `agents.reward` is the only reward signal, shared team rewards are broadcast to active slots. Because inactive slots are zeros, extracting the team scalar is not `agents.reward[0]` (slot 0 may be dead), use the active mask consumers already carry: `rewards[jnp.argmax(active_t)]`.
- **Nothing is zeroed for you at training time.** The environment zeros observations and rewards of inactive slots, but masking learning updates and bootstrapping with `active` is the consumer's job.
- **Value factorization is a deliberate exception.** Methods like QMIX may keep inactive agents in training with masked no-op actions, ignoring the transition-validity rule on purpose. Both patterns are expressible, neither is imposed.
