"""MLLM probe wrappers used by FlowGuard.

Each wrapper implements the ``MLLMProbe`` interface in ``base.py`` and exposes
two methods:

* ``probe(image, text, neutral_text)`` -> (P_t, P_v, P_mm) full distributions
* ``generate(image, text, max_new_tokens)`` -> str (for utility evaluation)

The ``load`` factory returns the wrapper appropriate for ``model_key``.
"""
from __future__ import annotations

from flowguard.models.base import MLLMProbe, ProbeOutputs, NEUTRAL_PROMPT

__all__ = ["MLLMProbe", "ProbeOutputs", "load", "NEUTRAL_PROMPT"]


def load(model_key: str, **kwargs) -> MLLMProbe:
    """Factory: dispatch to the wrapper matching ``model_key``.

    Recognized keys:
      - llava-1.5-7b
      - qwen2.5-vl-7b
      - gemma-3-4b
      - llama-3.1-70b-vl
      - gpt-4.1-mini
    """
    key = model_key.lower()
    if key.startswith("llava"):
        from flowguard.models.llava import LlavaProbe
        return LlavaProbe(model_key=key, **kwargs)
    if key.startswith("qwen"):
        from flowguard.models.qwen_vl import QwenVLProbe
        return QwenVLProbe(model_key=key, **kwargs)
    if key.startswith("gemma"):
        from flowguard.models.gemma import GemmaProbe
        return GemmaProbe(model_key=key, **kwargs)
    if key.startswith("llama-3.1-70b") or key.startswith("llama"):
        from flowguard.models.llama_vision import LlamaVisionProbe
        return LlamaVisionProbe(model_key=key, **kwargs)
    if key.startswith("gpt-") or key.startswith("openai"):
        from flowguard.models.gpt4_api import GPT4APIProbe
        return GPT4APIProbe(model_key=key, **kwargs)
    raise ValueError(f"Unknown model_key: {model_key}")
