"""LLaVA-1.5-7B probe wrapper.

We use the HuggingFace ``llava-hf`` port (``llava-hf/llava-1.5-7b-hf``) which
exposes the same architecture as the original ``liuhaotian/llava-v1.5-7b`` but
loads via ``LlavaForConditionalGeneration``. Only the next-token distribution
at the first decoding step is needed for FlowVector extraction.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from flowguard.models.base import MLLMProbe, NEUTRAL_PROMPT, ProbeOutputs
from flowguard.models._hf_common import (
    device_map_for,
    first_token_distribution,
    hf_cache_dir,
    load_dtype,
)

_HF_REPO = {
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
}


def _blank_image(size: int = 336) -> Image.Image:
    """Pure-zero RGB image used as a "no-image" placeholder for text-only probes.

    The LLaVA processor expects an image even when conditioning text-only, so
    we feed a blank canvas. Empirically this leaves the distribution on a
    text-only-equivalent regime; the appendix verifies the choice (Appendix A.5).
    """
    return Image.new("RGB", (size, size), color=(0, 0, 0))


class LlavaProbe(MLLMProbe):
    def __init__(self, model_key: str = "llava-1.5-7b", **_) -> None:
        super().__init__(model_key)
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
        )

        repo = _HF_REPO[model_key]
        self.processor = AutoProcessor.from_pretrained(
            repo, cache_dir=hf_cache_dir()
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo,
            torch_dtype=load_dtype(),
            device_map=device_map_for(repo),
            cache_dir=hf_cache_dir(),
        )
        self.model.eval()
        self.vocab_size = self.model.config.text_config.vocab_size

    # ------------------------------------------------------------------
    def _build_inputs(self, image: Image.Image | None, text: str) -> dict[str, torch.Tensor]:
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        img = image if image is not None else _blank_image()
        inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    def probe(
        self,
        image: Image.Image | None,
        text: str,
        neutral_text: str = NEUTRAL_PROMPT,
        decoding_steps: int = 1,
    ) -> ProbeOutputs:
        t0 = time.perf_counter()
        inputs_text = self._build_inputs(None, text)        # P_t  (blank image + Q)
        inputs_vis = self._build_inputs(image, neutral_text)  # P_v  (image + neutral Q)
        inputs_mm = self._build_inputs(image, text)         # P_mm (image + Q)

        p_t = first_token_distribution(self.model, inputs_text, decoding_steps)
        p_v = first_token_distribution(self.model, inputs_vis, decoding_steps)
        p_mm = first_token_distribution(self.model, inputs_mm, decoding_steps)

        return ProbeOutputs(
            p_text=p_t,
            p_vision=p_v,
            p_multimodal=p_mm,
            decoding_steps=decoding_steps,
            meta={"latency_s": time.perf_counter() - t0, "vocab_size": int(self.vocab_size)},
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | None,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        inputs = self._build_inputs(image, text)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = float(temperature)
        out = self.model.generate(**inputs, **kwargs)
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hidden_state(self, image: Image.Image | None, text: str) -> np.ndarray:
        """Final-layer hidden state at the last input token (Raw Embedding baseline)."""
        inputs = self._build_inputs(image, text)
        out = self.model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        return h.float().detach().to("cpu").numpy().reshape(-1)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
