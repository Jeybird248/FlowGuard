"""Common dataset abstractions and a registry-based ``load_split`` factory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from PIL import Image


@dataclass
class Sample:
    """A single multimodal evaluation sample."""

    sample_id: str
    image: Image.Image | None
    query: str
    label: str               # benign | unsafe | safe
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """Lazy iterable over Samples plus inspection metadata."""

    name: str
    iterator: Callable[[], Iterator[Sample]]
    expected_size: int | None = None
    label_distribution: str = ""

    def __iter__(self) -> Iterator[Sample]:
        return self.iterator()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., DatasetSplit]] = {}


def register(name: str) -> Callable[[Callable[..., DatasetSplit]], Callable[..., DatasetSplit]]:
    def deco(fn: Callable[..., DatasetSplit]) -> Callable[..., DatasetSplit]:
        _REGISTRY[name] = fn
        return fn
    return deco


def load_split(name: str, **kwargs: Any) -> DatasetSplit:
    """Look up a registered loader by name.

    Recognized names (matching paper Section 4.3):

      benign:
        - vqav2_train, vqav2_val
        - vizwiz_val, mossbench

      unsafe:
        - mmsb, mmsb_<category>
        - vlsafe
        - vlsu_unsafe, vlsu_safe
    """
    # Trigger registration side-effects.
    from flowguard.data import vqav2, mossbench, vizwiz, mmsafety, vlsafe, vlsu  # noqa: F401

    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset split: {name}. "
                       f"Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_splits() -> list[str]:
    from flowguard.data import vqav2, mossbench, vizwiz, mmsafety, vlsafe, vlsu  # noqa: F401
    return sorted(_REGISTRY)
