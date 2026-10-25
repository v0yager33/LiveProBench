"""M4 baseline (LongVA-based interruption/streaming model).

M4 is a LLaVA-NeXT/LongVA derivative: ``LlavaQwenForCausalLM`` with a SigLIP tower wired up
by the ``intersuit`` package inside the M4 repository, so it is loaded through that
repository's own builder rather than ``AutoModel``.

Two deliberate choices, both about staying faithful to the protocol rather than flattering
the baseline:

* Its ``LiveInfer`` streaming class is not reused.  That class owns a KV-cache session, a
  frame queue and its own speak/stay-silent threshold -- a second, different streaming
  policy.  Scoring it would measure M4's decoder policy, not the same polling protocol
  every other model here is measured under.  Instead the prefix is rebuilt per turn exactly
  as for the others.
* No silence token is added to its vocabulary.  M4 genuinely has none -- neither its
  ``added_tokens`` nor its repository defines a string for "say nothing", and the
  threshold its ``LiveInfer`` would have decided this with is commented out.  So unlike the
  other baselines its silence is *instructed* rather than trained, which the run records as
  ``native_silence_token: false``.  Its own system prompt is used as the base and the
  silence sentence appended, because that prompt on its own never tells the model it is
  allowed to stay quiet and would score 0 on every silence subtask for a reason that says
  nothing about M4.

M4 reads frames, so the media layer extracts one JPEG per chunk. Each frame is passed as a
single-frame ``video`` entry so the spatial pooling that makes long streams fit is applied,
one ``<image>`` token per chunk.

Its streamed turn carries no text, which is M4's own layout rather than a stripped-down one.
Its loop appends frame *embeddings* straight into the KV cache with no per-frame string at all,
and the ``(Current Time = Ns)`` lines in ``inference.py`` are built after the turn has already
been appended (``inference.py:110`` then ``:112``) -- they are what it prints, not what it
reads.  Its query turn is likewise bare: ``append_message(roles[0], '\n' + query)``.  Handing
it the corpus's ``<chunk>``/``<stream_info>`` wrapper instead would be showing it text no M4
checkpoint has seen, i.e. measuring instruction-following.  The cost is that M4, like AURA, is
not shown an explicit clock; ``docs/baselines.md`` records what that does and does not lose.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional

from .. import SILENCE
from ..conversation import Request
from ..media import AUDIO, IMAGE, VIDEO, Capabilities
from .base import Dialect, GenerationConfig, ModelError
from .local import LocalModel, add_repo_to_path, trim_completion

DEFAULT_CHECKPOINT = "/chatgpt_nas/dukaixuan.dkx/models/M4-LongVA-7B-Qwen2"
# The importable package lives one level inside the clone.
DEFAULT_REPO = "/chatgpt_nas/dukaixuan.dkx/M4/intersuit"

# The checkpoint's config.json names its vision tower ``checkpoints/clip-vit-large-patch14-336``
# -- a path relative to the M4 repo's own working directory, which resolves to nothing here and
# is then handed to the hub as a repo id.  The tower is loaded eagerly (``mm_tunable_parts``
# lists ``mm_vision_tower``, and all 391 of its tensors are in the checkpoint), so what this
# path supplies is the architecture and the preprocessor, both immediately overwritten by M4's
# own fine-tuned tower weights.  It still has to exist, or loading dies before the first row.
DEFAULT_VISION_TOWER = "/chatgpt_nas/dukaixuan.dkx/models/clip-vit-large-patch14-336"

# First sentence verbatim from the repository:
#   intersuit/local_demo/arguments_live.py:8-11  system_prompt
# The second sentence has no counterpart there -- see the module docstring. It reuses the
# dataset's ``<|silence|>`` for want of anything of M4's own, and ``native=False`` keeps that
# visible in every run record so this number is never read as a trained behaviour.
SYSTEM_PROMPT = (
    "A multimodal AI assistant is helping users with some activities."
    " Below is their conversation, interleaved with the list of video frames received by"
    " the assistant."
    f" At each frame the assistant decides whether to speak; to stay silent it outputs only"
    f" `{SILENCE}`."
)
# As with AURA, the streamed turn is M4's own -- which is to say wordless -- rather than a
# wrapper borrowed from the corpus (module docstring).  The benchmark's structure still holds
# around it: one turn per stream second, in order, the query as its own opening turn.
CHUNK_TURN = "{media}"
QUERY_TURN = "{text}"
DIALECT = Dialect(
    silence=SILENCE,
    system_prompt=SYSTEM_PROMPT,
    chunk_turn=CHUNK_TURN,
    query_turn=QUERY_TURN,
    native=False,
)


class M4(LocalModel):
    """M4 driven through the same per-turn polling protocol as every other model."""

    dialect = DIALECT

    default_name = "m4"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        repo: str = DEFAULT_REPO,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        conv_mode: str = "qwen_1_5",
        vision_tower: str = DEFAULT_VISION_TOWER,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            # Frames only: the vision tower takes preprocessed pixel values, and this
            # checkpoint has no audio encoder.
            capabilities=Capabilities(video=False, image=True, audio=False),
            generation=generation,
            display_name=display_name,
            device=device,
        )
        self.repo = add_repo_to_path(repo, "M4")
        self.attn_implementation = attn_implementation
        self.conv_mode = conv_mode
        self.vision_tower = vision_tower
        # What was asked for above; what was actually used is only known at load time.
        self.attn_used = ""

    def _load(self) -> None:
        try:
            import torch  # noqa: F401  - imported for its side effect of failing early
            from intersuit.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from intersuit.conversation import SeparatorStyle, conv_templates
            from intersuit.mm_utils import get_model_name_from_path, tokenizer_image_token
            from intersuit.model.builder import load_pretrained_model
        except ImportError as exc:
            raise ModelError(
                f"cannot import the M4 'intersuit' package from {self.repo}; "
                "install the M4 requirements into this environment"
            ) from exc

        self._image_token = DEFAULT_IMAGE_TOKEN
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._tokenize = tokenizer_image_token
        self._conv_templates = conv_templates
        self._separator_style = SeparatorStyle

        overwrite_config = {
            # Matches the repository's own demo: video frames go through 2x2 average pooling,
            # without which a 50-chunk prefix does not fit the context window.
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": "average",
            # Rewrites the unresolvable relative path; see DEFAULT_VISION_TOWER.
            "mm_vision_tower": self.vision_tower,
        }
        # M4 needs transformers 4.44 (its ``transformers.cache_utils`` imports were removed
        # later) and torch >= 2.5 (its module body imports ``flex_attention``), and a
        # flash-attn built for that pair may not be installable in that window.  Falling back
        # is better than not running the baseline at all, but it is written into the run
        # record rather than swallowed: sdpa is close to flash attention, not identical to it.
        attn = self.attn_implementation
        if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            attn = "sdpa"
        self.attn_used = attn
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.checkpoint,
            None,
            get_model_name_from_path(self.checkpoint),
            device_map=self.device,
            multimodal=True,
            attn_implementation=attn,
            overwrite_config=overwrite_config,
        )
        self.model.eval()

    # -- prompt building ---------------------------------------------------------------

    def _payload(self, request: Request):
        """Return (prompt, frame_paths) with one image token per frame, in order."""
        conversation = self._conv_templates[self.conv_mode].copy()
        frames: List[str] = []
        system_text = ""
        pending: List[str] = []

        for turn in request.turns:
            pieces = [turn.content] if isinstance(turn.content, str) else turn.content
            rendered: List[str] = []
            for piece in pieces:
                if isinstance(piece, str):
                    rendered.append(piece)
                elif piece.kind == IMAGE:
                    frames.append(piece.path)
                    rendered.append(self._image_token)
                elif piece.kind in (VIDEO, AUDIO):  # pragma: no cover - Capabilities filters
                    raise ModelError(f"m4 cannot consume {piece.kind} input")
            text = "".join(rendered)
            if turn.role == "system":
                system_text = text
            elif turn.role == "user":
                pending.append(text)
            else:
                conversation.append_message(conversation.roles[0], "\n".join(pending))
                conversation.append_message(conversation.roles[1], text)
                pending = []

        if system_text:
            # The template stores the whole ChatML system block, not just its body.
            conversation.system = f"<|im_start|>system\n{system_text}"
        conversation.append_message(conversation.roles[0], "\n".join(pending) if pending else "")
        conversation.append_message(conversation.roles[1], None)
        return conversation, conversation.get_prompt(), frames

    def generate(self, request: Request) -> str:
        # Before _payload: the conversation template and the image token index both come
        # from the repo modules that _load imports.
        self.ensure_loaded()
        import torch
        from PIL import Image

        conversation, prompt, frame_paths = self._payload(request)
        input_ids = self._tokenize(
            prompt, self.tokenizer, self._image_token_index, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        images = []
        tensors = []
        try:
            for path in frame_paths:
                if not os.path.exists(path):
                    raise ModelError(f"frame missing at {path}")
                image = Image.open(path).convert("RGB")
                images.append(image)
                # One single-frame entry per chunk, so one image token maps to one chunk.
                tensors.append(
                    self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"]
                    .to(self.model.device, dtype=torch.float16)
                )
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids,
                    images=tensors or None,
                    modalities=["video"] * len(tensors),
                    do_sample=False,
                    max_new_tokens=self.generation.max_tokens,
                    use_cache=True,
                )
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - one long row must not end the run
            raise ModelError(f"{self.name} generation failed: {exc}") from exc
        finally:
            for image in images:
                image.close()

        completion = self.tokenizer.decode(output[0], skip_special_tokens=False)
        separator = (
            conversation.sep
            if conversation.sep_style != self._separator_style.TWO
            else conversation.sep2
        )
        return trim_completion(completion, extra=(separator,) if separator else ())

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({
            "repo": self.repo,
            "conv_mode": self.conv_mode,
            "attn_implementation": self.attn_used or self.attn_implementation,
            "attn_implementation_requested": self.attn_implementation,
            "vision_tower_path": self.vision_tower,
            "vision_input": "frames",
            # M4 cannot be held to the shared pixel budget: its vision tower is a CLIP
            # ViT-L/14-336, so every frame is resized to a fixed 336x336 (112896 pixels,
            # under a fifth of LONGEST_EDGE) and then 2x2 average pooled.  Recorded rather
            # than silently omitted, because it is a real limit on what this row can be
            # compared against -- see docs/baselines.md.
            "pixel_budget": "fixed by architecture",
            "vision_tower": "clip-vit-large-patch14-336",
            "effective_pixels_per_frame": 336 * 336,
            "spatial_pool_stride": 2,
        })
        return info
