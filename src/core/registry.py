"""
Decorator-based component registry.

Provides a generic Registry class that allows registering and building
components (models, encoders, datasets, etc.) by name. This replaces
traditional Factory patterns with a more Pythonic, extensible approach.

Usage:
    ENCODER_REGISTRY = Registry("encoder")

    @ENCODER_REGISTRY.register("hand_cnn")
    class HandCNNEncoder(BaseEncoder):
        ...

    encoder = ENCODER_REGISTRY.build("hand_cnn", **kwargs)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A generic registry for mapping string names to classes.

    Supports decorator-based registration, config-driven instantiation,
    and introspection of registered components.

    Attributes:
        name: Human-readable name for this registry (e.g., "encoder").
    """

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: Descriptive name for this registry (used in error messages).
        """
        self._name = name
        self._registry: Dict[str, Type[Any]] = {}

    @property
    def name(self) -> str:
        """Return the registry name."""
        return self._name

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a class under the given name.

        Can be used as a decorator:
            @registry.register("my_component")
            class MyComponent: ...

        Args:
            name: The string key to register the class under.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If the name is already registered.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Cannot register '{name}' in '{self._name}' registry: "
                    f"already registered to {self._registry[name].__name__}."
                )
            self._registry[name] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class by name.

        Args:
            name: The registered name of the component.
            **kwargs: Arguments passed to the class constructor.

        Returns:
            An instance of the registered class.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in '{self._name}' registry. "
                f"Available: [{available}]"
            )
        cls = self._registry[name]
        return cls(**kwargs)

    def get(self, name: str) -> Type[Any]:
        """Retrieve a registered class without instantiating it.

        Args:
            name: The registered name of the component.

        Returns:
            The registered class.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in '{self._name}' registry. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def list_registered(self) -> list[str]:
        """Return a sorted list of all registered component names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._registry)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        items = ", ".join(sorted(self._registry.keys()))
        return f"Registry(name='{self._name}', items=[{items}])"


# ============================================================================
# Global registries for each component type
# ============================================================================

ENCODER_REGISTRY = Registry("encoder")
FUSION_REGISTRY = Registry("fusion")
TEMPORAL_REGISTRY = Registry("temporal")
DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("sampler")
LOSS_REGISTRY = Registry("loss")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
TRACKER_REGISTRY = Registry("tracker")
