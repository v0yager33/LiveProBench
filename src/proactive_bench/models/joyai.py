"""JoyAI-VL-Interaction.

An interactive vision-language assistant, evaluated here as a baseline.  Its own repository
(``/chatgpt_nas/dukaixuan.dkx/JoyAI-VL-Interaction``) ships a containerised service stack
with its own turn-taking policy; LiveProBench replaces that loop with its own protocol so
every model answers the same question with the same history.

The weights load through a ``Qwen3VLForConditionalGeneration`` architecture, which is why
stock vLLM serves them without the repository on ``PYTHONPATH``.  That is how it loads, not
what it is -- it shares an interface with AURA, nothing more.

This backend has no audio encoder, so its adapter applies the normal capability fallback.

It is the one baseline whose silence signal is not a ``<|...|>`` token at all: it answers
with ``</silence>`` or ``</response> <text>``, so both the prompt and the speak/silence rule
are translated into that dialect.

Its streamed user turn is translated too.  Both the training-data converter and the live
service write one the same way -- the query if there is one, then a bare ``<t seconds>``
marker, then one image per frame -- so the row's ``<chunk>``/``<stream_info>`` wrapper is
re-laid in that form.  Same frame, same time, spelled the way it was trained to read it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Dialect, GenerationConfig
from .openai_chat import VisionOnlyChatModel

DEFAULT_CHECKPOINT = "/chatgpt_nas/dukaixuan.dkx/models/JoyAI-VL-Interaction"
DEFAULT_REPO = "/chatgpt_nas/dukaixuan.dkx/JoyAI-VL-Interaction"
DEFAULT_PORT = 8032

# Verbatim from the repository:
#   services/webinfer/live_adapter.py:113  DEFAULT_SYSTEM_PROMPT_NO_DELEGATION
#   datasets/convert_data.py:249-251       assistant turns are "</response> {text}" / "</silence>"
# ``</silence>`` and ``</response>`` are tokens 151669 and 151670 in this checkpoint's
# added_tokens.json -- trained signals, not instructions it is following.
#
# The no-delegation variant is the right one of the two the repo offers. Its third action,
# ``</delegation> <question>``, hands the question to a background solver that does not exist
# here, and a turn spent delegating is neither silence nor an answer -- there would be no
# honest way to score it.
SILENCE_TAG = "</silence>"
RESPONSE_TAG = "</response>"
# Its streamed user turn, from the two places the repo builds one and agrees with itself:
#   datasets/convert_data.py:237-244       question, then f"<{sec:.1f} seconds>", then <image> xN
#   services/webinfer/live_adapter.py:1510-1523  query header + text, then f"<{marker}>", then
#                                                one image part per frame
# The marker's accepted spelling is fixed by TIME_RANGE_RE (live_adapter.py:93), which reads
# "<1.0 seconds>" and "<1.0s ~ 2.0s>"; the training converter used the first, so we do too.
QUERY_HEADER = "[User Query (IMPORTANT — follow this instruction)]"
CHUNK_TURN = "<{start} seconds>\n{media}"
QUERY_TURN = QUERY_HEADER + "\n{text}\n<{start} seconds>"
SYSTEM_PROMPT = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step you MUST choose exactly one of the following two actions:
**Stay silent** — output ONLY:
</silence>
Choose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.
**Speak** — output the token followed by a concise reply:
</response> Your reply here.
Choose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence.

## Important
There is NO delegation action. NEVER output </delegation> or hand off questions to any background solver."""
DIALECT = Dialect(
    silence=SILENCE_TAG,
    speak_prefix=RESPONSE_TAG,
    system_prompt=SYSTEM_PROMPT,
    chunk_turn=CHUNK_TURN,
    query_turn=QUERY_TURN,
    native=True,
)


class JoyAI(VisionOnlyChatModel):
    """JoyAI-VL-Interaction served over vLLM's OpenAI-compatible API."""

    default_name = "joyai"
    dialect = DIALECT

    def __init__(
        self,
        model_id: str = "",
        base_url: str = "",
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        workers: int = 4,
        vision_input: str = "frames",
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id or DEFAULT_CHECKPOINT,
            base_url=base_url or f"http://localhost:{DEFAULT_PORT}/v1",
            generation=generation,
            display_name=display_name or self.default_name,
            workers=workers,
            vision_input=vision_input,
            mm_processor_kwargs=mm_processor_kwargs,
        )
