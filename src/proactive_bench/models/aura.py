"""AURA.

A streaming vision-language assistant, evaluated here as a baseline.  Its own repository
(``/chatgpt_nas/dukaixuan.dkx/AURA``) drives a full streaming agent loop -- VAD, turn taking,
context management -- which LiveProBench deliberately replaces with its own protocol so
that every model is asked the same question with the same history.  What is measured is the
checkpoint's own willingness to stay quiet, not its controller's.

The weights load through a ``Qwen3VLForConditionalGeneration`` architecture, which is why
stock vLLM serves them without the repository on ``PYTHONPATH``.  That is how it loads, not
what it is.

This backend has no audio encoder, so its adapter applies the normal capability fallback.

Its silence token is ``<|silent|>`` -- not the ``<|silence|>`` the corpus is written in -- so
the prompt and the scoring are translated into AURA's own dialect.

Every string below is copied from AURA's own OmniMMI PA evaluation rather than from its
streaming demo (``AURA_bench_eval/OmniMMI/QWen3VL_online.py``), because PA is this benchmark's
counterpart task: one poll per streamed second, asking only whether to speak yet.  That script
answers what the demo cannot -- how AURA lays a streamed turn out when it is being *scored* --
and the answer is that it lays out nothing.  The turn is that second's media and no text at
all: no wrapper, no timestamp, because one segment per turn already carries the clock
positionally.  Its query turn is likewise bare text.

Four things in that script are the benchmark's to decide and are deliberately not copied: it
hands the server 1-second mp4 segments where every model here is fed the extracted frames, it
rebuilds each poll from at most the last 30 seconds where a row's whole prefix is replayed
here, it writes a hardcoded ``<|silent|>`` into every history turn where here the model's own
earlier answers are fed back, and its ack turn is a fixed "OK, I will let you know then."
where here the row's own ack is replayed.  Those are the protocol, identical for all six
models.

Two further things in it are not differences at all, having been adopted benchmark-wide: the
pixel budget it sends (``shortest_edge`` 4096 / ``longest_edge`` 602112, now ``base.py``'s
constants for every model) and the fact that its polling loop starts at the stream's second
second, never asking for a decision on the first -- which is what ``excluded_turns`` does here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Dialect, GenerationConfig
from .openai_chat import VisionOnlyChatModel

DEFAULT_CHECKPOINT = "/chatgpt_nas/dukaixuan.dkx/models/AURA"
DEFAULT_REPO = "/chatgpt_nas/dukaixuan.dkx/AURA"
DEFAULT_PORT = 8031

# Verbatim from AURA's own OmniMMI PA evaluation, so that it is asked for silence here in
# the words it is scored in there:
#   AURA_bench_eval/OmniMMI/QWen3VL_online.py:8        SYSTEM_PROMPT
#   AURA_bench_eval/OmniMMI/QWen3VL_online.py:147      "<|silent|>" written into history
#   AURA_bench_eval/OmniMMI/QWen3VL_online.py:134-159  the streamed turn: media, nothing else
#   AURA_bench_eval/OmniMMI/QWen3VL_online.py:115-123  the query turn: bare text
# ``<|silent|>`` is token 151669 in this checkpoint's added_tokens.json, i.e. a real token
# the model emits rather than a string we asked it to type.
#
# Its streaming demo (Qwen3_VL_online_streaming_v2_ContextManaged.py:135) spells the same
# prompt with ``'<|silent|>'`` in single quotes and one further sentence, "Respond in
# Chinese.".  The PA script's spelling is the one used here: that demo is a live product with
# a controller and a language requirement of its own, neither of which is what this benchmark
# scores, and PA is the task AURA's authors chose these words for.
SILENT_TEXT = "<|silent|>"
SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual "
    "context. Otherwise, output `<|silent|>` to signify silence."
)
# The media and nothing else, which is AURA's own layout for a scored streaming turn rather
# than a stripped-down one -- see the module docstring.  The benchmark's structure still holds
# around it: one turn per stream second, in order, the query as its own opening turn.  What
# AURA gives up relative to the models that print a clock is an explicit timestamp, which is
# the trade its own evaluation makes too.
CHUNK_TURN = "{media}"
QUERY_TURN = "{text}"
DIALECT = Dialect(
    silence=SILENT_TEXT,
    system_prompt=SYSTEM_PROMPT,
    chunk_turn=CHUNK_TURN,
    query_turn=QUERY_TURN,
    native=True,
)


class AURA(VisionOnlyChatModel):
    """AURA served over vLLM's OpenAI-compatible API."""

    default_name = "aura"
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
            # vLLM answers to whatever ``--served-model-name`` it was given, which the
            # launch is expected to set to the checkpoint path.
            model_id=model_id or DEFAULT_CHECKPOINT,
            base_url=base_url or f"http://localhost:{DEFAULT_PORT}/v1",
            generation=generation,
            display_name=display_name or self.default_name,
            workers=workers,
            vision_input=vision_input,
            mm_processor_kwargs=mm_processor_kwargs,
        )
