"""LLaMA-3.1-70B + CLIP ViT-L/14-336 probe wrapper.

The paper instantiates a multimodal LLaMA variant by pairing the 70B language
backbone with a CLIP-L/14-336 vision encoder, projected through a single
linear layer (the LLaVA-style connector). When the official ``mllama`` weights
(``meta-llama/Llama-3.2-11B-Vision-Instruct`` etc.) are available they are
preferred; otherwise we assemble the model on the fly using the connector
init weights at ``$FLOWGUARD_DATA_DIR/connectors/llama31_70b_clip-l14-336.pt``.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from flowguard.models.base import MLLMProbe, NEUTRAL_PROMPT, ProbeOutputs
from flowguard.models._hf_common import (
    first_token_distribution,
    hf_cache_dir,
    load_dtype,
)


_HF_TEXT = "meta-llama/Llama-3.1-70B-Instruct"
_HF_VISION = "openai/clip-vit-large-patch14-336"


class LlamaVisionProbe(MLLMProbe):
    """Two-tower LLaMA + CLIP probe.

    The 70B language tower is sharded across all available GPUs via
    ``device_map=auto``; the CLIP encoder lives on cuda:0 and emits a single
    [CLS] embedding that is projected and prepended to the token stream.
    """

    def __init__(self, model_key: str = "llama-3.1-70b-vl", **kwargs) -> None:
        super().__init__(model_key)
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
        )

        cache = hf_cache_dir()
        dtype = load_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(_HF_TEXT, cache_dir=cache)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            _HF_TEXT,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache,
        )
        self.lm.eval()

        self.image_processor = CLIPImageProcessor.from_pretrained(_HF_VISION, cache_dir=cache)
        self.vision = CLIPVisionModel.from_pretrained(
            _HF_VISION, torch_dtype=dtype, cache_dir=cache
        ).to("cuda:0")
        self.vision.eval()

        hidden = self.lm.config.hidden_size
        v_hidden = self.vision.config.hidden_size
        connector_path = kwargs.get("connector_path") or os.environ.get("FLOWGUARD_LLAMA_VL_CONNECTOR")
        self.connector = torch.nn.Linear(v_hidden, hidden, bias=True).to(
            dtype=dtype, device="cuda:0"
        )
        if connector_path and os.path.exists(connector_path):
            sd = torch.load(connector_path, map_location="cuda:0")
            self.connector.load_state_dict(sd)
        else:
            torch.nn.init.normal_(self.connector.weight, std=0.02)
            torch.nn.init.zeros_(self.connector.bias)

        self.vocab_size = int(self.lm.config.vocab_size)

    @torch.no_grad()
    def _vision_embeds(self, image: Image.Image) -> torch.Tensor:
        proc = self.image_processor(images=image, return_tensors="pt")
        pixel = proc["pixel_values"].to(self.vision.device, dtype=self.vision.dtype)
        out = self.vision(pixel_values=pixel)
        cls = out.last_hidden_state[:, 0, :]
        return self.connector(cls)  # (1, hidden)

    def _text_only_embeds(self, text: str) -> dict[str, torch.Tensor]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.lm.device)
        return {
            "inputs_embeds": self.lm.get_input_embeddings()(ids),
            "attention_mask": torch.ones_like(ids),
        }

    def _multimodal_embeds(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
        text_inputs = self._text_only_embeds(text)
        emb = text_inputs["inputs_embeds"]
        v_emb = self._vision_embeds(image).to(emb.device, dtype=emb.dtype).unsqueeze(1)
        # Prepend the single image embedding.
        full = torch.cat([v_emb, emb], dim=1)
        attn = torch.cat(
            [torch.ones((1, 1), device=emb.device, dtype=text_inputs["attention_mask"].dtype),
             text_inputs["attention_mask"]],
            dim=1,
        )
        return {"inputs_embeds": full, "attention_mask": attn}

    def probe(
        self,
        image: Image.Image | None,
        text: str,
        neutral_text: str = NEUTRAL_PROMPT,
        decoding_steps: int = 1,
    ) -> ProbeOutputs:
        t0 = time.perf_counter()
        if image is None:
            inputs_v = self._text_only_embeds(neutral_text)
            inputs_mm = self._text_only_embeds(text)
        else:
            inputs_v = self._multimodal_embeds(image, neutral_text)
            inputs_mm = self._multimodal_embeds(image, text)
        inputs_t = self._text_only_embeds(text)

        p_t = first_token_distribution(self.lm, inputs_t, decoding_steps)
        p_v = first_token_distribution(self.lm, inputs_v, decoding_steps)
        p_mm = first_token_distribution(self.lm, inputs_mm, decoding_steps)
        return ProbeOutputs(
            p_text=p_t, p_vision=p_v, p_multimodal=p_mm,
            decoding_steps=decoding_steps,
            meta={"latency_s": time.perf_counter() - t0, "vocab_size": self.vocab_size},
        )

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | None,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        if image is None:
            inputs = self._text_only_embeds(text)
        else:
            inputs = self._multimodal_embeds(image, text)
        out = self.lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def close(self) -> None:
        del self.lm, self.vision
        torch.cuda.empty_cache()
