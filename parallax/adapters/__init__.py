import typing

if typing.TYPE_CHECKING:
    from .brax import BraxAdapter as BraxAdapter
    from .gymnax import GymnaxAdapter as GymnaxAdapter

__all__ = ["BraxAdapter", "GymnaxAdapter"]


def __getattr__(name: str):
    if name == "BraxAdapter":
        from .brax import BraxAdapter

        return BraxAdapter
    if name == "GymnaxAdapter":
        from .gymnax import GymnaxAdapter

        return GymnaxAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
