# Changelog

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
