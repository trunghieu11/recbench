"""Class registries. Importing a module that uses the decorators fills these."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self) -> None:
        self.methods: dict[str, type] = {}
        self.metrics: dict[str, type] = {}
        self.datasets: dict[str, type] = {}

    def method(self, cls: type) -> type:
        self.methods[cls.spec.name] = cls
        return cls

    def metric(self, cls: type) -> type:
        self.metrics[cls.spec.name] = cls
        return cls

    def dataset(self, cls: type) -> type:
        self.datasets[cls.spec.name] = cls
        return cls

    def create_method(self, name: str) -> Any:
        if name not in self.methods:
            raise KeyError(f"Unknown method {name}. Registered: {sorted(self.methods)}")
        return self.methods[name]()

    def create_metric(self, name: str) -> Any:
        return self.metrics[name]()

    def create_dataset(self, name: str) -> Any:
        if name not in self.datasets:
            raise KeyError(f"Unknown dataset {name}. Registered: {sorted(self.datasets)}")
        return self.datasets[name]()


registry = Registry()


def register_method(cls: type[T]) -> type[T]:
    return registry.method(cls)


def register_metric(cls: type[T]) -> type[T]:
    return registry.metric(cls)


def register_dataset(cls: type[T]) -> type[T]:
    return registry.dataset(cls)


def ensure_loaded() -> Registry:
    from recbench import datasets as _datasets  # noqa: F401
    from recbench import metrics as _metrics  # noqa: F401
    from recbench import methods as _methods  # noqa: F401

    return registry
