import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PRNGKeyArray

from .env import Env
from .struct import EnvState, Timestep


class VmapWrapper:
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(
        self,
        *,
        key: PRNGKeyArray,
        state: EnvState | None = None,
        timestep: Timestep | None = None,
        done: Bool[Array, "..."] | None = None,
    ) -> tuple[EnvState, Timestep]:
        if done is None or state is None or timestep is None:
            return jax.vmap(self.env.reset)(key=key)
        keys = jax.random.split(key, done.shape[0])
        new_state, new_timestep = jax.vmap(self.env.reset)(key=keys)
        out_state: EnvState = jax.tree.map(
            lambda n, o: jnp.where(done, n, o), new_state, state
        )
        out_timestep: Timestep = jax.tree.map(
            lambda n, o: jnp.where(done, n, o), new_timestep, timestep
        )
        return out_state, out_timestep

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]:
        return jax.vmap(self.env.step)(state, action)


class TimeLimit:
    def __init__(self, env: Env, max_steps: int):
        self.env = env
        self.max_steps = max_steps
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, key: PRNGKeyArray) -> tuple[EnvState, Timestep]:
        return self.env.reset(key=key)

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, Timestep]:
        state, timestep = self.env.step(state, action)
        truncation = timestep.truncation | (state.step >= self.max_steps)
        timestep = Timestep(
            observation=timestep.observation,
            reward=timestep.reward,
            termination=timestep.termination,
            truncation=truncation,
            info=timestep.info,
        )
        return state, timestep
