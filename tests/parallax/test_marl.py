"""Contract tests for the multi-agent (MARL) protocol."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from conftest import assert_trees_close, assert_trees_equal
from parallax.core import Agents, MARLState
from parallax.spaces import Box, Discrete
from parallax.wrappers import TimeLimit, VmapWrapper


class LastAgentStandingEnv:
    """Three agents die at steps 2, 4 and 6, termination when all are dead.

    Active agents add their action value to a running score each step and
    receive reward 1 + action. With `minimal=True` the optional fields
    (`action_mask`, `global_observation`) are `None`.
    """

    num_agents = 3
    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=100.0, shape=())

    def __init__(self, minimal: bool = False) -> None:
        self.minimal = minimal
        self.death_steps = jnp.array([2, 4, 6], dtype=jnp.int32)

    def reset(self, *, key):
        return MARLState(
            env_state={"count": jnp.float32(0.0), "scores": jnp.zeros(self.num_agents)},
            agents=Agents(
                observation=jnp.zeros(self.num_agents),
                active=jnp.ones(self.num_agents, dtype=bool),
                reward=jnp.zeros(self.num_agents),
                action_mask=None if self.minimal else jnp.ones((self.num_agents, 2), dtype=bool),
            ),
            termination=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
            step_count=jnp.int32(0),
            key=key,
            global_observation=None if self.minimal else jnp.zeros(self.num_agents + 1),
        )

    def step(self, state, action):
        acted = state.agents.active
        count = state.env_state["count"] + 1.0
        scores = state.env_state["scores"] + action.astype(jnp.float32) * acted
        step_count = state.step_count + 1
        active = step_count < self.death_steps
        mask = jnp.stack([jnp.ones(self.num_agents, dtype=bool), active], axis=1)
        return MARLState(
            env_state={"count": count, "scores": scores},
            agents=Agents(
                observation=jnp.where(active, count, 0.0),
                active=active,
                reward=(1.0 + action.astype(jnp.float32)) * acted,
                action_mask=None if self.minimal else mask,
            ),
            termination=~active.any(),
            truncation=jnp.bool_(False),
            info={},
            step_count=step_count,
            key=state.key,
            global_observation=None if self.minimal else jnp.concatenate([count[None], scores]),
        )


ONES = jnp.ones(3, dtype=jnp.int32)


def rollout(env, num_steps, action=ONES):
    state = env.reset(key=jax.random.key(0))
    for _ in range(num_steps):
        state = env.step(state, action)
    return state


class TestStructure:
    def test_reset_returns_marl_state(self):
        state = LastAgentStandingEnv().reset(key=jax.random.key(0))
        assert isinstance(state, MARLState)
        assert isinstance(state.agents, Agents)

    def test_reset_all_active(self):
        state = LastAgentStandingEnv().reset(key=jax.random.key(0))
        assert jnp.all(state.agents.active)
        assert not state.done

    def test_agent_leaves_have_leading_dim(self):
        state = rollout(LastAgentStandingEnv(), 1)
        for leaf in jtu.tree_leaves(state.agents):
            assert leaf.shape[0] == 3

    def test_mdp_fields_are_scalar(self):
        state = rollout(LastAgentStandingEnv(), 1)
        assert state.termination.shape == ()
        assert state.truncation.shape == ()
        assert state.done.shape == ()

    def test_global_observation(self):
        state = rollout(LastAgentStandingEnv(), 2)
        assert state.global_observation is not None
        assert state.global_observation.shape == (4,)
        assert state.global_observation[0] == 2.0


class TestContract:
    def test_inactive_actions_ignored(self):
        """Rule 1: garbage in inactive slots does not affect the transition."""
        env = LastAgentStandingEnv()
        state = rollout(env, 2)  # agent 0 dead
        a = env.step(state, jnp.array([0, 1, 1], dtype=jnp.int32))
        b = env.step(state, jnp.array([1, 1, 1], dtype=jnp.int32))
        assert_trees_equal(a, b)

    def test_inactive_observations_zero(self):
        """Rule 2: dead agents observe zeros."""
        state = rollout(LastAgentStandingEnv(), 3)  # agent 0 died at step 2
        assert state.agents.observation[0] == 0.0
        assert jnp.all(state.agents.observation[1:] == 3.0)

    def test_inactive_rewards_zero(self):
        """Rule 3: slots inactive at action time carry zero reward."""
        state = rollout(LastAgentStandingEnv(), 3)
        assert state.agents.reward[0] == 0.0
        assert jnp.all(state.agents.reward[1:] == 2.0)

    def test_dying_step_carries_reward(self):
        """Rule 3: the step an agent dies on is an active step with reward."""
        state = rollout(LastAgentStandingEnv(), 2)
        assert not state.agents.active[0]  # just died
        assert state.agents.reward[0] == 2.0  # but acted this step

    def test_inactive_mask_has_legal_action(self):
        """Rule 4: dead agents keep at least one legal action."""
        state = rollout(LastAgentStandingEnv(), 6)
        assert state.agents.action_mask is not None
        assert not jnp.any(state.agents.active)
        assert jnp.all(state.agents.action_mask.any(axis=1))

    def test_active_marks_next_actors(self):
        """Rule 5: active in state_t masks the actions used by step t+1."""
        env = LastAgentStandingEnv()
        state = rollout(env, 3)  # agent 0 inactive for steps 3 onward
        assert jnp.array_equal(state.env_state["scores"], jnp.array([2.0, 3.0, 3.0]))

    def test_termination_only_when_env_says_so(self):
        """Rule 6: the env sets termination when the game ends, not before."""
        env = LastAgentStandingEnv()
        state = rollout(env, 5)
        assert not state.termination  # agent 2 still alive
        state = env.step(state, ONES)
        assert state.termination
        assert not jnp.any(state.agents.active)


class TestTeamReward:
    """Shared team rewards live in agents.reward, broadcast to active slots."""

    def test_extraction_needs_active_mask(self):
        """The team scalar is read from a slot that was active, not slot 0."""
        env = LastAgentStandingEnv()
        prev = rollout(env, 2)  # agent 0 dead
        state = env.step(prev, ONES)
        assert state.agents.reward[0] == 0.0  # naive slot 0 read is wrong
        team_reward = state.agents.reward[jnp.argmax(prev.agents.active)]
        assert team_reward == 2.0


class TestBootstrapMask:
    def test_die_on_truncation_step(self):
        """Agent dies on the same step the env truncates, no special casing."""
        env = TimeLimit(LastAgentStandingEnv(), max_steps=4)
        state = rollout(env, 4)  # agent 1 dies at step 4, truncation at step 4
        assert state.truncation
        assert not state.termination
        bootstrap = state.agents.active & ~state.termination
        assert jnp.array_equal(bootstrap, jnp.array([False, False, True]))

    def test_no_bootstrap_on_termination(self):
        state = rollout(LastAgentStandingEnv(), 6)
        assert state.termination
        bootstrap = state.agents.active & ~state.termination
        assert not jnp.any(bootstrap)


class TestWrappers:
    """SARL wrappers work on MARL envs unchanged."""

    def test_time_limit(self):
        env = TimeLimit(LastAgentStandingEnv(), max_steps=3)
        state = rollout(env, 3)
        assert state.truncation
        assert not state.termination

    def test_vmap_reset(self):
        env = VmapWrapper(LastAgentStandingEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        assert state.agents.active.shape == (4, 3)
        assert state.agents.reward.shape == (4, 3)

    def test_vmap_step(self):
        env = VmapWrapper(LastAgentStandingEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        state = env.step(state, jnp.ones((4, 3), dtype=jnp.int32))
        assert state.agents.observation.shape == (4, 3)
        assert state.global_observation is not None
        assert state.global_observation.shape == (4, 4)

    def test_vmap_selective_reset(self):
        env = VmapWrapper(LastAgentStandingEnv(), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        actions = jnp.ones((4, 3), dtype=jnp.int32)
        state = env.step(state, actions)
        state = env.step(state, actions)  # agent 0 dead in all envs

        done = jnp.array([True, False, False, False])
        state = env.reset(key=jax.random.key(1), state=state, done=done)
        assert jnp.all(state.agents.active[0])
        assert not state.agents.active[1, 0]
        assert state.step_count[0] == 0
        assert state.step_count[1] == 2

    def test_wrapper_stack(self):
        env = VmapWrapper(TimeLimit(LastAgentStandingEnv(), max_steps=4), num_envs=4)
        state = env.reset(key=jax.random.key(0))
        for _ in range(4):
            state = env.step(state, jnp.ones((4, 3), dtype=jnp.int32))
        assert jnp.all(state.truncation)


class TestPytree:
    def test_jit_step(self):
        env = LastAgentStandingEnv()
        state = env.reset(key=jax.random.key(0))
        s1 = env.step(state, ONES)
        s2 = jax.jit(env.step)(state, ONES)
        assert_trees_close(s1, s2)

    def test_jit_with_none_fields(self):
        env = LastAgentStandingEnv(minimal=True)
        state = env.reset(key=jax.random.key(0))
        s1 = env.step(state, ONES)
        s2 = jax.jit(env.step)(state, ONES)
        assert s2.agents.action_mask is None
        assert s2.global_observation is None
        assert_trees_close(s1, s2)

    def test_replace(self):
        state = LastAgentStandingEnv().reset(key=jax.random.key(0))
        state = replace(state, truncation=jnp.bool_(True))
        assert state.done

    def test_scan_rollout(self):
        env = LastAgentStandingEnv()
        state = env.reset(key=jax.random.key(0))

        def step_fn(state, _):
            state = env.step(state, ONES)
            return state, state.agents.active

        state, actives = jax.lax.scan(step_fn, state, None, length=6)
        assert state.termination
        assert jnp.array_equal(actives.sum(axis=0), jnp.array([1, 3, 5]))
