import typing

if typing.TYPE_CHECKING:
    from .brax import BraxAdapter as BraxAdapter
    from .gymnax import GymnaxAdapter as GymnaxAdapter
    from .mjx import MJXAdapter as MJXAdapter

__all__ = ["BraxAdapter", "GymnaxAdapter", "MJXAdapter"]

_EXTRAS = {
    "BraxAdapter": "brax",
    "GymnaxAdapter": "gymnax",
    "MJXAdapter": "mjx",
}


def __getattr__(name: str):
    if name not in _EXTRAS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        if name == "BraxAdapter":
            from .brax import BraxAdapter

            return BraxAdapter
        if name == "GymnaxAdapter":
            from .gymnax import GymnaxAdapter

            return GymnaxAdapter
        if name == "MJXAdapter":
            from .mjx import MJXAdapter

            return MJXAdapter
    except ImportError:
        extra = _EXTRAS[name]
        raise ImportError(
            f"{name} requires extra dependencies. Install with:\n"
            f"  pip install parallax-rl[{extra}]\n"
            f"or install all adapters:\n"
            f"  pip install parallax-rl[adapters]"
        ) from None
