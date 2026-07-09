# Changelog

## v0.3.1

- Add `stack_space` to build spaces with a leading `num_agents` dimension, exported from `parallax`
- `Space` protocol now declares a `shape` property, implemented by all spaces
- `PyTreeSpace.shape` returns a pytree of the leaf space shapes, mirroring `sample`
- `MultiDiscrete` supports multi-dimensional `actions_per_dim`
- `MultiBinary` accepts a tuple `n` for multi-dimensional shapes

## v0.3.0

- Add multi-agent protocol: `MARLEnv`, `MARLVectorEnv`, `MARLState`, and `Agents`
- Episode termination and truncation stay scalar, per-agent lifecycle is tracked with `agents.active`
- Per-agent rewards in `agents.reward`, shared team rewards broadcast to active slots
- Optional `action_mask` and `global_observation` fields for action masking and centralised critics
- Existing wrappers (`TimeLimit`, `VmapWrapper`) work on MARL environments unchanged
- Wrappers are generic over the state type, fully typed for both protocols and subclassed states

## v0.2.5

- Fix dtype mismatch in Gymnax adapter: `info['discount']` is now consistently `float32` in both `reset()` and `step()`

## v0.2.4

- MJX adapter extracts `episode_length` from environment config for built-in truncation

## v0.2.3

- Add MuJoCo Playground (MJX) adapter
- Add optional dependency groups: `parallax-rl[brax]`, `parallax-rl[gymnax]`, `parallax-rl[mjx]`, `parallax-rl[adapters]`
- Adapter imports show install instructions when dependencies are missing

## v0.2.2

- Brax adapter strips `AutoResetWrapper` and `EpisodeWrapper`, handling truncation internally

## v0.2.1

- Add `VectorEnv` protocol for vectorised environments with selective reset
- Add GitHub Actions workflow for automatic PyPI publishing
- Lazily import adapters

## v0.2.0

- Add Gymnax and Brax adapters
- Simplify `State` dataclass by removing lazy properties
- Simplify `Env` protocol

## v0.1.0

- Initial release
- `Env` protocol and `State` dataclass
- `VmapWrapper`, `AutoResetWrapper`, and `TimeLimit` wrappers
- Basic test suite
