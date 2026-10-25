"""The contract every model adapter implements.

An adapter answers exactly one question: given a rebuilt conversation prefix, what does
this model emit for the next assistant turn?  Everything about *which* prefixes get built
and how the answers are scored lives outside, so adding a baseline never touches the
scoring code.

Two properties beyond ``generate`` matter to the harness:

``capabilities``
    What the model's own input pipeline accepts.  The media layer converts a row's media
    into that shape -- a backend that takes no audio is handed the frames alone, and one
    that wants a chunk mp4 rather than the extracted frames would be handed that.

``max_concurrency``
    An upper bound the runner will not exceed.  A hosted endpoint tolerates several
    in-flight rows; a locally loaded checkpoint holds one GPU and must stay serial.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .. import SILENCE
from ..conversation import Request
from ..media import Capabilities


class ModelError(RuntimeError):
    """The model could not answer this prompt.

    Raised for both transport failures and hard server-side rejections (a row exceeding
    ``--limit-mm-per-prompt`` or the context window).  The runner records such a row as
    ``status != "ok"`` and moves on, because these are properties of individual long rows
    rather than of the run.
    """


# The pixel budget every model is held to, so that a score difference is a difference in
# behaviour and not in how much of the frame each model was shown.  Both numbers are areas
# in pixels, not edge lengths: 602112 = 768 x 784.  This is the budget AURA's own streaming
# evaluations use, and the one StreamingBench / OVO-Bench / OmniMMI are run at here, so the
# numbers stay comparable across benchmarks.
#
# Each checkpoint's own default is far larger (AURA ships 65536/16777216, MOSS-VL
# 4096/16777216), so leaving it unset does not mean "the same" -- it means each model picks
# its own resolution.  Every adapter that can accept a budget is given this one.
SHORTEST_EDGE = 4096
LONGEST_EDGE = 602112


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    retries: int = 3
    timeout: float = 600.0


@dataclass(frozen=True)
class Dialect:
    """How one model's own repository writes a streaming turn.

    Four things differ between the baselines and not one of them is a property of the task:
    the string a model emits to stay silent, the prefix its repo puts in front of spoken
    text, the system prompt its repo pairs with that vocabulary, and the way its repo lays a
    streamed chunk out as a user turn.  The dataset is written in our model's dialect, so
    replaying it verbatim would instruct a baseline in a vocabulary it was never trained on
    and then score it on a token it has no reason to emit.  AURA's silence token is
    ``<|silent|>`` and JoyAI's is ``</silence>``; asking either for ``<|silence|>`` measures
    instruction-following, not the trained behaviour the benchmark is about.  Each adapter
    declares its own dialect, copied from its own repository, and the harness translates the
    conversation into it.

    What is deliberately *not* per-model: the frames, the audio track, the pixel budget, the
    turn structure, and which turns get polled.  Those are the benchmark, and holding them
    fixed is the only reason two models' numbers can be put side by side.

    The chunk's *timestamp* is part of that per-model wording rather than something the
    benchmark imposes.  Two repos write one (JoyAI ``<1.0 seconds>``, MOSS-VL
    ``<|time_start|>1.0 seconds<|time_end|>``) and three write none, being driven by a live
    clock where elapsed time is implicit in how many frames have arrived -- so their chunk
    turn here is the media and nothing else.  That is not free: replay drops the seconds an
    assistant spent speaking, so the k-th chunk turn is not always the k-th second, on 600 of
    the 2599 published rows.  Those 600 are all in the two pure-silence subtasks, where the
    offset exists *because* the reference already spoke and those spoken turns are visible in
    the prefix; the signal a model needs there is its own answered turns, not the clock.  The
    alternative -- lending a quiet repo the corpus's wrapper so that it gets a printed time --
    shows it text no checkpoint of it has seen, which is the thing this class exists to avoid.
    """

    #: What the model emits to stay silent.
    silence: str = SILENCE
    #: What its repo puts in front of spoken text, if anything (JoyAI: ``</response>``).
    speak_prefix: str = ""
    #: Control tokens its repo hides when rendering output, and which therefore do not count
    #: as having said anything (MOSS-VL: ``<|response|>``, ``<|assistant|>``, ``<go>`` and the
    #: round markers, all stripped by its ``render_output``).  Empty for a model whose repo
    #: strips nothing but the silence string.
    hidden: Tuple[str, ...] = ()
    #: Its repo's system prompt. Required, like the two templates below: the strings the
    #: corpus was authored in are one model's (``LiveFlow_omni.py``), and a field left empty
    #: used to inherit them, which is how a baseline would end up instructed to emit a token
    #: it was never trained to produce.
    system_prompt: str = ""
    #: How its repo writes one streamed chunk as a user turn. ``{media}`` is where the chunk's
    #: frames go; ``{start}``/``{end}`` are its bounds in seconds to one decimal, available to
    #: a template that prints a clock and simply unused by one that does not.
    chunk_turn: str = ""
    #: Likewise for the opening query turn: ``{text}`` is the question, ``{start}`` its time.
    query_turn: str = ""
    #: True when ``silence`` is a real token in this checkpoint's vocabulary rather than a
    #: string we asked a model to type. Recorded per run because it is the difference
    #: between measuring a trained signal and measuring instruction-following.
    native: bool = True

    def spoke(self, text: Optional[str]) -> bool:
        """This model's own speak-or-silent rule.

        Two shapes, both taken from the models' own inference code rather than invented
        here.  With a speak prefix, the earliest marker wins and unmarked text counts as
        speech -- JoyAI's ``normalize_model_output``.  Without one, the silence string and any
        tokens the repo hides when rendering are stripped, and a turn is silent iff nothing
        else remains -- MOSS-VL's ``render_output``, which tests ``"<|silence|>" in chunk``,
        replaces the occurrences away, and then does the same for its control markers before
        deciding there is anything left to print.

        Stripping *every* occurrence rather than testing the whole reply for equality is not
        laxness about stray tokens: a MOSS-VL speaking turn ends by emitting the silence
        token, so ``text<|silence|>`` is its ordinary shape for speech.  What this rule cannot
        do is decide a turn from a reply that continues past that closing token -- the leading
        silence would be stripped and the continuation read as speech -- so each adapter is
        responsible for cutting a poll off at its own turn boundary before asking here.
        """
        if text is None:
            return False
        raw = str(text).strip()
        if not raw:
            return False
        if self.speak_prefix:
            silent_at = raw.find(self.silence)
            spoken_at = raw.find(self.speak_prefix)
            if silent_at == -1 and spoken_at == -1:
                return True
            if spoken_at == -1:
                return False
            return silent_at == -1 or spoken_at < silent_at
        for token in self.hidden:
            raw = raw.replace(token, "")
        return raw.replace(self.silence, "").strip() != ""

    def as_silence(self) -> str:
        """A silent assistant turn, for replaying history in this model's own words."""
        return self.silence

    def as_speech(self, text: str) -> str:
        """A spoken assistant turn, prefixed the way this model's repo prefixes one."""
        body = str(text or "").strip()
        if not self.speak_prefix:
            return body
        if body.startswith(self.speak_prefix):
            return body
        return f"{self.speak_prefix} {body}".strip()

    def describe(self) -> Dict[str, Any]:
        return {
            "silence": self.silence,
            "speak_prefix": self.speak_prefix or None,
            "hidden_tokens": list(self.hidden) or None,
            "native_silence_token": self.native,
            "prints_chunk_time": "{start}" in self.chunk_turn,
        }


class ProactiveModel(ABC):
    """One model under evaluation."""

    #: Whether this checkpoint has an audio encoder at all -- a fact about the weights, not
    #: about how a given run was configured.  ``capabilities.audio`` answers the different
    #: question of whether audio is being fed *this* run, and goes False under ``--audio
    #: off``; without this constant an ablation run would be indistinguishable from a model
    #: that never could hear, which is precisely the distinction the ablation exists to draw.
    has_audio_encoder: bool = False

    #: How this model's own repository writes a streaming turn. Every adapter declares its
    #: own; this bare default carries no wording at all, so an adapter that forgets fails on
    #: its first poll with the field named rather than quietly borrowing another model's.
    dialect: Dialect = Dialect()

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded on every prediction; also the results subdirectory name."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    @property
    def max_concurrency(self) -> int:
        return 1

    @abstractmethod
    def generate(self, request: Request) -> str:
        """Return the next assistant turn's text.

        Must preserve special tokens verbatim: the silence token is the signal being
        measured, so a decoder that strips it turns every silent turn into an empty string
        and silently changes what is scored.
        """

    def generate_with_diagnostics(self, request: Request) -> Tuple[str, Dict[str, Any]]:
        """Return the turn's text plus whatever else the model decided it on.

        For a model that answers in words the text *is* the whole decision, so the default
        reports nothing extra.  ROMA is different: its speak-or-stay-silent choice is a
        probability read off a gate head, and the text is only a consequence of that
        probability having crossed a threshold.  Recording the probability per turn is what
        lets a finished run be rescored at another threshold instead of re-run.

        Whatever a model returns here is stored under the prediction's ``diagnostics`` key,
        nested rather than merged, so a new field can never shadow ``response`` or any other
        part of the scored record.
        """
        return self.generate(request), {}

    def describe(self) -> Dict[str, Any]:
        """Run-provenance fields for the summary file.

        The dialect is recorded on every run because it is not derivable from the
        transcript: seeing ``</silence>`` in a prediction does not say whether the model was
        asked for it in its own words or merely happened to produce it.
        """
        return {"model": self.name, "backend": type(self).__name__, "dialect": self.dialect.describe()}

    def close(self) -> None:
        """Release held resources. Safe to call more than once."""
