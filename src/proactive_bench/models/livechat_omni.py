"""Our model: Qwen3-Omni OPSD behind the resident vLLM server.

This is the only adapter with an audio encoder, and therefore the only one that can be run
both with and without audio on the same rows.  The pixel budget and fps are passed
per-request as ``mm_processor_kwargs`` so that every model in the suite -- including the
vision-only baselines, whose servers are started with the same numbers -- sees the same
number of frames at the same resolution.  That budget has to be spelled *twice*; see
``extra_body``.

Like every other adapter it is fed extracted frames plus a per-chunk audio track rather
than the chunk mp4.  It is the one model that *could* read the mp4 directly, and feeding it
that way while the baselines get frames would make the visual stream differ between our
model and theirs on the one axis the benchmark is meant to hold fixed.

The corpus was authored in this model's dialect, so the three strings below are also the
wording the delivery rows carried -- which is why they live here and not in the protocol
layer.  They are this checkpoint's own rendering of the problem, exactly as AURA's prompt is
AURA's, and they were taken out of the published annotation for that reason.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .. import SILENCE
from ..conversation import Request
from ..media import Capabilities
from .base import LONGEST_EDGE, SHORTEST_EDGE, Dialect, GenerationConfig
from .openai_chat import OpenAIChatModel

DEFAULT_BASE_URL = "http://localhost:8028/v1"
DEFAULT_MODEL_ID = (
    "/chatgpt_nas/dukaixuan.dkx/LiveChatOmni/eval_logs/omnimmi_pa_0831_100step/"
    "qwen3omni_opsd_0831_100step_vllmwrap"
)

# Matches the resident server's --media-io-kwargs '{"video":{"num_frames":-1,"fps":1}}':
# one frame per second, i.e. exactly one frame per 1s chunk.
DEFAULT_FPS = 1.0
DEFAULT_SHORTEST_EDGE = SHORTEST_EDGE
DEFAULT_LONGEST_EDGE = LONGEST_EDGE

# This checkpoint's training dialect, and therefore the wording every delivery row was
# annotated in.  The step that built the published set pins all three against those rows:
# filling ``{media}`` with the single ``<video>`` placeholder the rows carried reproduces
# their messages byte for byte, over all 2599 of them, which is the check that converting
# them to annotation lost nothing.  Editing any of these strings will fail that build.
CHUNK_TURN = (
    "<chunk>\n{media}\n<stream_info>start_time: {start} s\n"
    "end_time: {end} s</stream_info>\n</chunk>"
)
QUERY_TURN = "<query>\n{text}\n<stream_info>query_time: {start} s</stream_info>\n</query>"
SYSTEM_PROMPT = (
    "# Role\n"
    "You are an AI assistant that processes real-time streaming video. In each subsequent "
    "turn of the conversation, you will receive temporal chunks from a real-time video "
    "stream. Each turn represents one continuous segment of the video stream.\n"
    "\n"
    "# Core Task\n"
    "For each video stream chunk in every interaction turn, decide whether to remain silent "
    "or respond.\n"
    "\n"
    f"- If you remain silent, output only `{SILENCE}`.\n"
    "- If you respond, do so naturally at the appropriate moment."
)
DIALECT = Dialect(
    silence=SILENCE,
    system_prompt=SYSTEM_PROMPT,
    chunk_turn=CHUNK_TURN,
    query_turn=QUERY_TURN,
)


class LiveChatOmni(OpenAIChatModel):
    """Qwen3-Omni OPSD served by the project's vLLM wrapper."""

    default_name = "LiveFlow_omni"
    send_part_uuid = True
    has_audio_encoder = True
    dialect = DIALECT

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        base_url: str = DEFAULT_BASE_URL,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        workers: int = 4,
        fps: float = DEFAULT_FPS,
        shortest_edge: int = DEFAULT_SHORTEST_EDGE,
        longest_edge: int = DEFAULT_LONGEST_EDGE,
        audio: bool = True,
        vision_input: str = "frames",
    ) -> None:
        if vision_input not in ("video", "frames"):
            raise ValueError(f"vision_input must be 'video' or 'frames', got {vision_input!r}")
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            # Under 'frames' the media layer hands over one JPEG per chunk plus that chunk's
            # extracted wav, so the mp4 is never sent even though this backend could take it.
            capabilities=Capabilities(
                video=vision_input == "video",
                image=True,
                audio=bool(audio),
            ),
            generation=generation,
            display_name=display_name,
            workers=workers,
        )
        self.vision_input = vision_input
        self.fps = float(fps)
        self.shortest_edge = int(shortest_edge)
        self.longest_edge = int(longest_edge)

    def extra_body(self, request: Request) -> Dict[str, Any]:
        body = super().extra_body(request)
        body["mm_processor_kwargs"] = {
            "fps": self.fps,
            # The same budget under both spellings, because the server applies each to only
            # one kind of part and silently drops the other.  Measured on :8028 with one 4K
            # frame and one chunk mp4, reading usage.prompt_tokens:
            #
            #                  no kwargs   size only   min/max only   both
            #   image part       8171        8171          587         587
            #   video part        731         587          731         587
            #
            # So sending only ``size`` -- as this adapter did until the pixel budget was
            # audited -- left every extracted frame at its native resolution, up to 14x the
            # budget for a 4K row, while the baselines were correctly capped.  That is the
            # one axis the benchmark exists to hold fixed, and it also overflowed the 32768
            # context on four rows.  Both spellings coexist without error and each still
            # caps its own part, so both are sent rather than branching on vision_input:
            # a branch would silently uncap again the next time the input path changes.
            "size": {"shortest_edge": self.shortest_edge, "longest_edge": self.longest_edge},
            "min_pixels": self.shortest_edge,
            "max_pixels": self.longest_edge,
            # Only meaningful when a video part is actually being sent.  With frames the
            # audio arrives as its own part, so asking the processor to lift a track out of
            # a video it never received would be a claim about the request that is not true.
            "use_audio_in_video": bool(request.use_audio) and self.vision_input == "video",
        }
        return body

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({
            "vision_input": self.vision_input,
            "fps": self.fps,
            "shortest_edge": self.shortest_edge,
            "longest_edge": self.longest_edge,
            # Which spellings the budget was actually sent under, not just its numbers.  A
            # run.json that records the numbers alone cannot distinguish a capped run from
            # the uncapped ones this adapter produced while sending only ``size``.
            "pixel_budget_keys": ["size", "min_pixels", "max_pixels"],
            # Two separate facts: the checkpoint can hear, and whether it was let to this
            # run.  Reporting only the second would make an --audio off run read as a
            # vision-only model rather than as the ablation it is.
            "has_audio_encoder": self.has_audio_encoder,
            "audio_fed": self.capabilities.audio,
        })
        return info
