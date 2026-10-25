"""Shared plumbing for baselines loaded into this process.

Two baselines cannot be served by vLLM -- MOSS-VL registers its own ``moss_vl``
architecture through ``trust_remote_code``, and M4 is a LongVA/LLaVA derivative whose
vision tower is wired up by its own repository code.  Both therefore hold a GPU inside the
harness, which has two consequences the runner needs to know about:

* ``max_concurrency`` is 1.  One process, one checkpoint, one in-flight prompt.
* the repository has to be importable.  Each adapter puts its own repo on ``sys.path``
  lazily, inside ``__init__``, so that running only the HTTP backends never imports a
  baseline's dependencies.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, Optional

from ..media import Capabilities
from .base import GenerationConfig, ModelError, ProactiveModel

# Every terminator a chat template can emit.  Stripped from completions decoded with
# ``skip_special_tokens=False``, which is the only way to keep the silence token intact.
TERMINATORS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "</s>",
    "<|end|>",
    "<pad>",
)


def add_repo_to_path(repo: str, label: str) -> str:
    """Prepend a baseline repository to ``sys.path`` and return the resolved path."""
    resolved = os.path.abspath(repo)
    if not os.path.isdir(resolved):
        raise ModelError(f"{label} repository not found at {resolved}")
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved


def trim_completion(text: str, extra: Iterable[str] = ()) -> str:
    """Drop terminator tokens from a completion, keeping every other special token.

    The silence token must survive verbatim -- it *is* the measurement -- so decoding
    cannot use ``skip_special_tokens=True`` and the terminators are removed by name
    instead.
    """
    out = str(text)
    for token in (*TERMINATORS, *extra):
        out = out.replace(token, "")
    return out.strip()


class LocalModel(ProactiveModel):
    """A checkpoint loaded in-process; subclasses implement ``generate``.

    Weights are loaded on the first ``generate`` rather than in ``__init__`` so that
    ``--check`` and a mistyped argument cost nothing: a 7B checkpoint takes minutes and a
    GPU to load, and a sweep that probes several backends before running them should not
    pay for that twice.
    """

    def __init__(
        self,
        checkpoint: str,
        capabilities: Capabilities,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        device: str = "cuda",
    ) -> None:
        if not os.path.isdir(checkpoint):
            raise ModelError(f"checkpoint not found at {checkpoint}")
        self.checkpoint = checkpoint
        self._capabilities = capabilities
        self.generation = generation or GenerationConfig()
        self._display_name = display_name or self.default_name
        self.device = device
        self._loaded = False

    default_name = "local"

    def _load(self) -> None:
        """Populate ``self.model`` and friends. Called once, before the first generate."""
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def probe(self) -> str:
        """Validate what can be checked without loading weights."""
        return f"{self.checkpoint} (not loaded yet)"

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def max_concurrency(self) -> int:
        return 1

    def describe(self) -> Dict[str, Any]:
        # Extends rather than replaces, for the same reason as the served transport: the
        # dialect comes from the base and a rebuilt dict would silently lose it.
        info = super().describe()
        info.update({
            "checkpoint": self.checkpoint,
            "device": self.device,
            "max_tokens": self.generation.max_tokens,
            # A property of the weights, and whether this run fed them.  Every model reports
            # both so run.json has one schema, and so an --audio off run never reads as a
            # model that had no ears to begin with.
            "has_audio_encoder": self.has_audio_encoder,
            "audio_fed": self.capabilities.audio,
        })
        return info

    def close(self) -> None:
        self._loaded = False
        for attribute in ("model", "processor", "tokenizer", "image_processor"):
            if getattr(self, attribute, None) is not None:
                setattr(self, attribute, None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - teardown must not mask a run's real error
            pass
