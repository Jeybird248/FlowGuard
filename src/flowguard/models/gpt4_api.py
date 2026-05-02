"""GPT-4.1-mini partial-distribution probe.

The OpenAI Chat Completions API exposes per-token ``logprobs`` with
``top_logprobs`` up to 20 candidates. We treat the visible tail as a truncated
distribution and renormalize via ``flowguard.utils.distributions.renormalize_topk``;
``compute_flowvector_topk`` then unions the three top-k tails before computing
the FlowVector.

This wrapper does not return full ``|V|`` distributions — its ``probe`` method
returns three top-k dicts that the caller passes to ``compute_flowvector_topk``.
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from flowguard.models.base import MLLMProbe, NEUTRAL_PROMPT, ProbeOutputs


def _png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class GPT4APIProbe(MLLMProbe):
    """Top-k logprob probe for OpenAI vision models.

    ``ProbeOutputs.p_text`` etc. carry a length-(K) numpy array of probabilities
    over the unioned top-k token set; ``meta["topk_text"]`` etc. carry the
    original token-string keys for downstream alignment.
    """

    def __init__(
        self,
        model_key: str = "gpt-4.1-mini",
        api_model_id: str | None = None,
        top_logprobs: int = 20,
        timeout: float = 60.0,
        **_,
    ) -> None:
        super().__init__(model_key)
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for GPT4APIProbe")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.api_model_id = api_model_id or model_key
        self.top_logprobs = int(top_logprobs)
        self.vocab_size = -1  # not exposed via API

    # ------------------------------------------------------------------
    def _query_topk(self, image: Image.Image | None, text: str) -> dict[str, float]:
        """Return ``{token_string: logprob}`` for the first generated token."""
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": _png_data_url(image)}})
        content.append({"type": "text", "text": text})
        resp = self.client.chat.completions.create(
            model=self.api_model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        choice = resp.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            return {}
        first = choice.logprobs.content[0]
        topk = {first.token: float(first.logprob)}
        for entry in (first.top_logprobs or []):
            topk[entry.token] = float(entry.logprob)
        return topk

    def probe(
        self,
        image: Image.Image | None,
        text: str,
        neutral_text: str = NEUTRAL_PROMPT,
        decoding_steps: int = 1,
    ) -> ProbeOutputs:
        if decoding_steps != 1:
            raise NotImplementedError(
                "Multi-step probing requires per-token logprobs; the API only "
                "returns the first generated token reliably for k=1."
            )
        t0 = time.perf_counter()
        topk_t = self._query_topk(None, text)
        topk_v = self._query_topk(image, neutral_text)
        topk_mm = self._query_topk(image, text)

        union = sorted(set(topk_t) | set(topk_v) | set(topk_mm))

        def _expand(dct: dict[str, float]) -> np.ndarray:
            obs = sum(np.exp(v) for v in dct.values())
            unobs = max(0.0, 1.0 - float(obs))
            missing = [tok for tok in union if tok not in dct]
            per_missing = unobs / max(len(missing), 1)
            arr = np.array(
                [float(np.exp(dct[tok])) if tok in dct else per_missing for tok in union],
                dtype=np.float64,
            )
            s = arr.sum()
            return arr / s if s > 0 else arr

        return ProbeOutputs(
            p_text=_expand(topk_t),
            p_vision=_expand(topk_v),
            p_multimodal=_expand(topk_mm),
            decoding_steps=1,
            meta={
                "latency_s": time.perf_counter() - t0,
                "topk_text": topk_t,
                "topk_vision": topk_v,
                "topk_multimodal": topk_mm,
                "union_size": len(union),
            },
        )

    def generate(
        self,
        image: Image.Image | None,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": _png_data_url(image)}})
        content.append({"type": "text", "text": text})
        resp = self.client.chat.completions.create(
            model=self.api_model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
