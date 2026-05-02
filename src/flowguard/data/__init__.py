"""Dataset loaders for FlowGuard.

Each loader yields ``Sample(image, query, label, meta)`` records. The image
field is a ``PIL.Image.Image`` (or ``None`` for text-only samples) and ``label``
is one of ``"benign"`` / ``"unsafe"`` / ``"safe"``. Categorical metadata
(harm category, original benchmark id) is stored in ``meta``.
"""
from __future__ import annotations

from flowguard.data.base import Sample, DatasetSplit, load_split

__all__ = ["Sample", "DatasetSplit", "load_split"]
