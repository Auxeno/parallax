<div align="center">

  <h1> Parallax </h1>

  <h3>A JAX Reinforcement Learning Protocol</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Why Parallax?

JAX RL environments need pure functions and immutable state, but there is no standard interface for them. Parallax defines a minimal `reset`/`step` contract so agents, wrappers and training loops work with any environment that follows it.

- **For JAX RL users**: write agents, rollouts and training loops once, swap environments without changing code
- **For Gymnasium users**: the same concepts (`reset`, `step`, observation, reward) as pure functions, so everything works with `jit`, `vmap` and `scan`

Parallax is a protocol, not a framework. No base class, no registration: an environment satisfies the protocol by having the right methods.

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

## The Protocol

Conventional RL environments are stateful: `env.step()` mutates the environment in place. JAX needs pure functions and immutable data, so Parallax splits the two:

**Env** is stateless. It has two pure functions, `reset` and `step`, and no internal state.

**State** is a JAX pytree that holds all the data. Every call to `reset` or `step` returns a new one:

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

## Building an Environment

Implement `reset` and `step`, each returning a `State` with all fields computed:

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

`env_state` is your raw environment data and can be any JAX pytree. The other fields are derived from it in `reset` and `step`.

## Wrappers

Wrappers compose to add functionality:

```python
from parallax import TimeLimit, VmapWrapper

env = VmapWrapper(TimeLimit(GridWorld(), max_steps=200), num_envs=128)
state = env.reset(key=jax.random.key(0))
state = env.step(state, actions)

# Terminal observations are still in state, reset only the finished envs
state = env.reset(key=reset_key, state=state, done=state.done)
```

- `VmapWrapper` steps `num_envs` environments in parallel with `jax.vmap` and adds selective reset
- `TimeLimit` truncates episodes after `max_steps`
- `AutoResetWrapper` restarts episodes as they end, replacing terminal observations with reset ones
- `Wrapper` is the base class for custom wrappers

Selective reset (`state=`, `done=`) keeps terminal observations available for value bootstrapping. The arguments pass through wrapper stacks to the vector env inside.

Wrappers forward unknown attributes to the wrapped env at runtime. Statically visible are the spaces, `num_agents` and, on `VmapWrapper`, `num_envs`.

## Multi-Agent Environments

Parallax separates two events that multi-agent code must treat differently: the episode ending and an individual agent dying. `MARLState` keeps `termination` and `truncation` as episode-level scalars and tracks per-agent lifecycle with an `active` mask:

```python
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

Agents occupy slots `0..num_agents - 1` and are shape-homogeneous: per-agent data stacks on a leading `num_agents` dimension, and `action_space`/`observation_space` describe that stacked view, built from a per-agent space with `stack_space`. Heterogeneous agents pad to the largest shape and express legality through `action_mask`.

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

Everything a learning algorithm needs is a one-liner from these fields:

```python
valid = state_t.agents.active                               # transition is real, use it in the loss
bootstrap = state_t1.agents.active & ~state_t1.termination  # died or terminated, no bootstrap
rewards = state_t1.agents.reward                            # pairs with state_t.agents.active
```

An agent that dies on the step the episode truncates needs no special casing: it stops bootstrapping through `active` while surviving agents bootstrap from the terminal observation.

Teams are not a protocol concept, slots are. Broadcast each team's reward to its members' slots in `agents.reward`: all slots equal is fully cooperative, `+1`/`-1` across two blocks of slots is two-team zero-sum. Team membership is environment metadata, subclass `Agents` to expose it.

The episode-level fields match the single-agent `State`, so wrappers like `TimeLimit` and `VmapWrapper` work on multi-agent environments unchanged.

### Building a Multi-Agent Environment

Three agents race to position 10. An agent that finishes becomes inactive, and the episode terminates when all have finished:

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from parallax import Agents, MARLState, Space, spaces, stack_space


class Race:
    num_agents = 3
    action_space: Space = stack_space(spaces.Discrete(2), num_agents)
    observation_space: Space = stack_space(spaces.Box(0.0, 10.0, ()), num_agents)

    def reset(self, *, key: PRNGKeyArray) -> MARLState:
        positions = jnp.zeros(self.num_agents)
        return MARLState(
            env_state=positions,
            agents=Agents(
                observation=positions,
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
        acted = state.agents.active
        positions = state.env_state + action * acted       # inactive actions are ignored
        active = positions < 10.0                          # finished agents become inactive
        finished_now = acted & ~active
        return MARLState(
            env_state=positions,
            agents=Agents(
                observation=jnp.where(active, positions, 0.0),  # zeros for inactive agents
                active=active,
                reward=jnp.where(finished_now, 1.0, 0.0),       # the finishing step carries reward
            ),
            termination=~active.any(),
            truncation=jnp.bool_(False),
            info={},
            step_count=state.step_count + 1,
            key=state.key,
        )
```

The spaces lead with `num_agents` (`action_space.shape == (3,)`), inactive slots are masked out of the transition and zeroed in observations and rewards, and the environment sets `termination` itself when the last agent finishes.

## Adapters

Use existing JAX RL environments through adapters:

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

Adapters map foreign reset/step APIs to the Parallax protocol. Brax and MJX adapters read the episode length from the wrapped environment and set `truncation` themselves, and Brax's built-in auto-reset is stripped to preserve terminal observations. Gymnax does not distinguish truncation from termination, so all episode endings are reported as `termination`.

## Custom State Fields

Subclass `State` to add fields. For example, an action mask:

```python
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass
class MaskedState(State):
    action_mask: Bool[Array, "4"]
```

Return it from your env's `reset` and `step`:

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

The protocols are generic in the state type: this env satisfies `Env[MaskedState]`, wrappers preserve the state type, and `state.action_mask` stays visible to type checkers through a full wrapper stack. Bare `Env` means `Env[State]`. The same pattern applies to `MARLState` and `Agents`.

## Collecting Experience

Use `jax.lax.scan` for vectorised rollouts. Selective resets capture terminal observations before resetting finished environments, which value bootstrapping needs:

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
- **Shapes widen under `VmapWrapper`, spaces do not.** A scalar `reward` becomes `(num_envs,)` and multi-agent `(num_agents,)` fields become `(num_envs, num_agents)`, but the spaces keep describing the unbatched environment.
- **Wrapper attribute forwarding is runtime-only.** Statically, wrappers expose the spaces, `num_agents` and `num_envs`. Anything else reached through a wrapper resolves via `__getattr__` and types as `Any`.

**Multi-agent**

- **The agent count is static.** `num_agents` is a trace-time constant and an upper bound. Agents can die, and the protocol does not forbid re-activation (respawns), but the slot count never changes.
- **Spaces lead with `num_agents`.** Per-agent network sizes come from the trailing dims: `observation_space.shape[-1]` for features, `Discrete.n` for action count.
- **The reward/active pairing is off by one by design.** Rewards in the state returned by `step` pair with the `active` mask of the state passed in. Scan-collected trajectories must respect this, see the timing diagram above.
- **There is no MDP-level reward.** `agents.reward` is the only reward signal, shared team rewards are broadcast to active slots. Because inactive slots are zeros, extracting the team scalar is not `agents.reward[0]` (slot 0 may be dead), use the active mask consumers already carry: `rewards[jnp.argmax(active_t)]`.
- **Nothing is zeroed for you at training time.** The environment zeros observations and rewards of inactive slots, but masking learning updates and bootstrapping with `active` is the consumer's job.
- **Value factorization is a deliberate exception.** Methods like QMIX may keep inactive agents in training with masked no-op actions, ignoring the transition-validity rule on purpose. Both patterns are expressible, neither is imposed.
