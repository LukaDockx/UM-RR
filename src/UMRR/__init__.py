"""UM-RR: uncoupled modes about a relaxed internal-rotation path."""

__version__ = "0.1.0"
__all__ = [
    "InitialConditionConfig",
    "InitialConditionGenerator",
    "InitialConditionResult",
]


def __getattr__(name):
    if name == "InitialConditionConfig":
        from .config import InitialConditionConfig

        return InitialConditionConfig
    if name in {"InitialConditionGenerator", "InitialConditionResult"}:
        from .sampler import InitialConditionGenerator, InitialConditionResult

        return {
            "InitialConditionGenerator": InitialConditionGenerator,
            "InitialConditionResult": InitialConditionResult,
        }[name]
    raise AttributeError(name)
