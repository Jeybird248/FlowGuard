"""Qwen2.5-VL-7B-Instruct probe wrapper.

Uses ``transformers.Qwen2VLForConditionalGeneration`` (the 2.5 series shares
the 2.0 modeling code with a different config). The chat template is rendered
via ``processor.apply_chat_template``.
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
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",
}


class QwenVLProbe(MLLMProbe):
    def __init__(self, model_key: str = "qwen2.5-vl-7b", **_) -> None:
        super().__init__(model_key)
        from transformers import AutoProcessor, AutoModelForCausalLM

        repo = _HF_REPO[model_key]
        self.processor = AutoProcessor.from_pretrained(repo, cache_dir=hf_cache_dir())
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=load_dtype(),
            device_map=device_map_for(repo),
            cache_dir=hf_cache_dir(),
            trust_remote_code=True,
        )
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

    def _build_inputs(self, image: Image.Image | None, text: str) -> dict[str, torch.Tensor]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def probe(
        self,
        image: Image.Image | None,
        text: str,
        neutral_text: str = NEUTRAL_PROMPT,
        decoding_steps: int = 1,
    ) -> ProbeOutputs:
        t0 = time.perf_counter()
        p_t = first_token_distribution(
            self.model, self._build_inputs(None, text), decoding_steps
        )
        p_v = first_token_distribution(
            self.model, self._build_inputs(image, neutral_text), decoding_steps
        )
        p_mm = first_token_distribution(
            self.model, self._build_inputs(image, text), decoding_steps
        )
        return ProbeOutputs(
            p_text=p_t, p_vision=p_v, p_multimodal=p_mm,
            decoding_steps=decoding_steps,
            meta={"latency_s": time.perf_counter() - t0, "vocab_size": int(self.vocab_size)},
        )

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | None,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        inputs = self._build_inputs(image, text)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def hidden_state(self, image: Image.Image | None, text: str) -> np.ndarray:
        inputs = self._build_inputs(image, text)
        out = self.model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        return h.float().detach().to("cpu").numpy().reshape(-1)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
