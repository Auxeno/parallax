<div align="center">

  <h1> 🪞 Parallax </h1>
  
  <h3>A Reinforcment Learning API</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Parallax is a reinforcement learning API focusing on:

- Default to vectorised environments
- No automatic resetting, more explicit and cleaner API for truncation support
- Agents can be individually removed from environments, to keep observations paralellisable, 

#### Issues

- Dynamic observations
    - Since agents are fixed, agent-related observations are always of a fixed size
    - If agents are missing, we zero them and return a mask

- Dynamic agents
    - Must pre-specify a number of agents
    - 

--- 

## API

Mirrors PettingZoo parallel:

```python

env.agent_types = {
    "agent_0": 0,
    "agent_1": 0,
    "agent_2": 1,
    "agent_3": ...
}

observations, infos = parallel_env.reset(seed=42)

observations, rewards, terminations, truncations, infos = env.step(actions)

observations = {
    "agent_0": {
        "vector":      shape=(num_envs, obs_dim),                 dtype=float,
        "pixel":       shape=(num_envs, height, width, channels), dtype=float,
        "set_0":       shape=(num_envs, set_size, obs_dim),       dtype=float,
        "set_0_mask":  shape=(num_envs, set_size),                dtype=bool,
        "set_1":       shape=(num_envs, set_size, obs_dim),       dtype=float,
        "set_1_mask":  shape=(num_envs, set_size),                dtype=bool,
         ...
        "action_mask": shape(num_envs, action_dim),              dtype=bool
    },
    "agent_1": ...
}

actions = {
    "agent_0": shape=(num_envs, num_actions), dtype=int,
    "agent_1": ...
}

rewards = {
    "agent_0": shape(num_envs), dtype=float,
    "agent_1": ...
}

terminations = {
    "agent_0": shape(num_envs), dtype=bool,
    "agent_1": ...
}

truncations = {
    "agent_0": shape(num_envs), dtype=bool,
    "agent_1": ...
}

```