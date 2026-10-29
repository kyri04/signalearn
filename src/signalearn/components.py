from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = ["Model"]


@dataclass(eq=False)
class Model:
    model: Any
    inputs: Mapping
    task: str
    classes: Any = None
    output: str | None = None
    outputs: Mapping | None = None
    loss: Any = None
    optimizer: Callable | None = None
    epochs: int = 1
    batch_size: int = 64
    shuffle: bool = True
    seed: int | None = None
    device: str | None = None
    trainer: Callable | None = None
    factory: Callable | None = None
    factory_params: Mapping | None = None

    def __post_init__(self):
        from ._internal.models import configure_model

        configure_model(self)
