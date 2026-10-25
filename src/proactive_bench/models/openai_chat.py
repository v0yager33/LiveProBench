"""Shared OpenAI-compatible chat backend.

LiveChatOmni, AURA and JoyAI are all reached through a vLLM OpenAI server, so the request
building, retry policy and per-thread client caching live once here and each concrete model
only declares its endpoint, its capabilities and its ``extra_body``.

:class:`VisionOnlyChatModel` adds the little that AURA and JoyAI share -- no audio encoder,
and a visual stream that can arrive as video or as frames.  It is a base class, not a
model: each baseline gets its own module named after itself.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..conversation import Request, Turn
from ..media import AUDIO, IMAGE, VIDEO, Capabilities, MediaRef
from .base import LONGEST_EDGE, SHORTEST_EDGE, GenerationConfig, ModelError, ProactiveModel

# Retryable transport hiccups vs. a request this server will never accept.  Retrying the
# latter just multiplies the wait before the row is recorded as failed.
FATAL_MARKERS = (
    "limit-mm-per-prompt",
    "at most",
    "maximum context length",
    "longer than the maximum",
    "reduce the length",
    "does not support",
)


def as_file_url(path: str) -> str:
    return Path(path).resolve().as_uri()


def _is_fatal(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in FATAL_MARKERS)


class OpenAIChatModel(ProactiveModel):
    """A model served over an OpenAI-compatible ``/chat/completions`` endpoint."""

    # Our vLLM wrapper keys its multimodal cache on a per-part ``uuid``; stock vLLM builds
    # rejects unknown part fields, so each subclass opts in.
    send_part_uuid: bool = False

    def __init__(
        self,
        model_id: str,
        base_url: str,
        capabilities: Capabilities,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        api_key: str = "EMPTY",
        workers: int = 4,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self._capabilities = capabilities
        self.generation = generation or GenerationConfig()
        self._display_name = display_name or self.default_name
        self._workers = max(1, int(workers))
        self._local = threading.local()

    default_name = "openai_chat"

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def max_concurrency(self) -> int:
        return self._workers

    def describe(self) -> Dict[str, Any]:
        # Extends rather than replaces: the base contributes the dialect, which is the only
        # record of whether the model was addressed in its own words, and rebuilding the dict
        # here would drop it from every served baseline's run.json without any error.
        info = super().describe()
        info.update({
            "model_id": self.model_id,
            "base_url": self.base_url,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
        })
        return info

    # -- request building ------------------------------------------------------------

    def media_part(self, ref: MediaRef, tag: str) -> Dict[str, Any]:
        url = as_file_url(ref.path)
        if ref.kind == VIDEO:
            part: Dict[str, Any] = {"type": "video_url", "video_url": {"url": url}}
        elif ref.kind == IMAGE:
            part = {"type": "image_url", "image_url": {"url": url}}
        elif ref.kind == AUDIO:
            part = {"type": "audio_url", "audio_url": {"url": url}}
        else:  # pragma: no cover - MediaRef validates its own kind
            raise ModelError(f"unsupported media kind {ref.kind!r}")
        if self.send_part_uuid:
            part["uuid"] = f"proactive-bench-{tag}-{uuid.uuid4().hex}"
        return part

    def turn_payload(self, turn: Turn, index: int) -> Dict[str, Any]:
        if isinstance(turn.content, str):
            return {"role": turn.role, "content": turn.content}
        parts: List[Dict[str, Any]] = []
        for position, piece in enumerate(turn.content):
            if isinstance(piece, str):
                if piece:
                    parts.append({"type": "text", "text": piece})
            else:
                parts.append(self.media_part(piece, f"{index}-{position}"))
        # A turn whose only content was a dropped audio placeholder would otherwise become
        # an empty part list, which the server rejects.
        if not parts:
            parts.append({"type": "text", "text": ""})
        return {"role": turn.role, "content": parts}

    def messages(self, request: Request) -> List[Dict[str, Any]]:
        return [self.turn_payload(turn, index) for index, turn in enumerate(request.turns)]

    def extra_body(self, request: Request) -> Dict[str, Any]:
        """Server-specific decode options.

        ``skip_special_tokens=False`` is not optional anywhere: with the default the
        silence token is stripped and every silent turn arrives as an empty string, which
        is indistinguishable from a truncated generation.
        """
        return {"skip_special_tokens": False}

    # -- transport -------------------------------------------------------------------

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ModelError("the openai package is required for HTTP backends") from exc
            client = self._local.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.generation.timeout,
                max_retries=0,
            )
        return client

    def generate(self, request: Request) -> str:
        payload = self.messages(request)
        extra_body = self.extra_body(request)
        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, self.generation.retries) + 1):
            try:
                response = self._client().chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    max_tokens=self.generation.max_tokens,
                    temperature=self.generation.temperature,
                    extra_body=extra_body,
                )
                if not response or not response.choices:
                    raise RuntimeError("response carried no choices")
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("response carried no content")
                return str(content).strip()
            except Exception as exc:  # noqa: BLE001 - every failure mode is retried alike
                last_error = exc
                if _is_fatal(str(exc)):
                    break
                if attempt < self.generation.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise ModelError(f"{self.name} request failed: {last_error}")

    def probe(self) -> List[str]:
        """Model ids the endpoint reports, so a typo fails before a long run starts."""
        try:
            return [item.id for item in self._client().models.list().data]
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"cannot reach {self.base_url}: {exc}") from exc


class VisionOnlyChatModel(OpenAIChatModel):
    """Base class for a served baseline with no audio encoder.

    Not a model in its own right -- AURA and JoyAI each subclass it and keep their own
    module, checkpoint and port.  Only the two things they genuinely share live here.

    ``vision_input`` decides *how* the visual stream is delivered:

    ``frames`` (default)
        One extracted JPEG per chunk.  Costs an ffmpeg pass, but is what every model in the
        suite receives, works against a server without video support, and is the layout the
        rows themselves are migrating to.

    ``video``
        One ``video_url`` part per chunk, i.e. the chunk mp4 handed to the server's own
        decoder.  Requires the server to be started with ``--limit-mm-per-prompt.video``
        high enough for the longest row.  Kept for comparing against server-side decoding;
        not the scored configuration.

    The audio track is unavailable to this backend. The baseline is shown as much of each
    supplied row as it can read.
    """

    # Stock vLLM validates content parts strictly and rejects the extra ``uuid`` key that
    # our own wrapper uses for multimodal caching.
    send_part_uuid = False

    def __init__(
        self,
        model_id: str,
        base_url: str,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        workers: int = 4,
        vision_input: str = "frames",
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if vision_input not in ("video", "frames"):
            raise ValueError(f"vision_input must be 'video' or 'frames', got {vision_input!r}")
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            capabilities=Capabilities(
                video=vision_input == "video",
                image=True,
                audio=False,
            ),
            generation=generation,
            display_name=display_name,
            workers=workers,
        )
        self.vision_input = vision_input
        # Left empty by default: the pixel budget and fps belong on the server's own
        # --mm-processor-kwargs, and an unknown per-request key makes vLLM reject the call.
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})

    def extra_body(self, request: Request) -> Dict[str, Any]:
        body = super().extra_body(request)
        if self.mm_processor_kwargs:
            body["mm_processor_kwargs"] = dict(self.mm_processor_kwargs)
        return body

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({
            "vision_input": self.vision_input,
            # Same pair every model reports, so run.json has one schema: this class is the
            # vision-only transport, so there are no ears to feed either way.
            "has_audio_encoder": self.has_audio_encoder,
            "audio_fed": self.capabilities.audio,
            "mm_processor_kwargs": self.mm_processor_kwargs or None,
            # What the server is expected to have been started with.  The harness cannot read
            # a vLLM server's --mm-processor-kwargs back out, so this records the budget the
            # score is only comparable at; the launch has to pass these same constants.
            "expected_server_pixel_budget": {
                "shortest_edge": SHORTEST_EDGE,
                "longest_edge": LONGEST_EDGE,
                "set_by": "--mm-processor-kwargs at serve time",
            },
        })
        return info


def parse_mm_processor_kwargs(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("mm_processor_kwargs must be a JSON object")
    return value
