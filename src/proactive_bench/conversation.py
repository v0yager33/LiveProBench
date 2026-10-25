"""Assembling one model's prompt for one polled turn.

The published set is annotation, not messages: a video, which second each stream chunk shows,
what the user asked, and what the reference assistant did on every turn.  This module turns
that into an actual conversation, once per poll and once per model.

The protocol is polling, not teacher-forced decoding.  To ask what a model would emit at the
turn following chunk ``k``, the conversation is rebuilt from the start through chunk ``k`` and
sent as a fresh request.  Its assistant prefix is normally the annotation's own reference
behaviour and not the model's earlier answers: each poll is one independent measurement, so a
miss earlier in a row cannot change the question a later turn is asked.  The one exception is
the running-count mode, whose reference turns *are* the answer and which therefore replays the
model's own replies instead -- see :func:`build_request`.

How far back that rebuild reaches is bounded: only the last :data:`CONTEXT_WINDOW_CHUNKS`
seconds of stream are in the prompt, the window advancing with the poll.  See that constant
for why the bound exists and what survives it.

Composition is per model, in the model's own dialect (:class:`models.base.Dialect`): its
system prompt, its silence token, its repo's way of laying a streamed chunk out.  What is
*not* per model is the multimodal content -- the same frames and the same audio in the same
order, at the same pixel budget.  That is the benchmark.  The chat *structure* built here --
one user turn per second, the query as its own opening turn -- is the shape most of these
repos read, and it is deliberately not held equal: an adapter whose checkpoint streams in a
different shape refolds these turns into that shape itself, which is what ``models/moss_vl.py``
does with its single open assistant turn.  A repository that opens its stream with the
question attached to the first frames still gets the question as its own turn here, and the
first chunk turn is left out of every subtask's scored targets for exactly that reason.

No model's wording lives in this module.  Every string a model is shown -- its system prompt,
its silence token, its chunk and query wrappers -- is declared in that model's own adapter,
including our model's: the corpus was authored in its dialect, which makes those strings its
property rather than the protocol's (``models/LiveFlow_omni.py``).  This module knows only
which turns exist, in what order, and where a template wants its media.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .annotation import Annotation, Chunk
from .annotation import Turn as AnnotationTurn
from .media import Capabilities, ChunkMedia, MediaPlan, MediaRef

#: Where a dialect's chunk template wants the chunk's frames.
MEDIA_SLOT = "{media}"

#: How many seconds of stream end up in one poll's prompt, counted back from the polled chunk.
#:
#: A row of this set can be four minutes long, and replaying all of it would ask a streaming
#: checkpoint a question no streaming deployment ever asks: the live systems these models come
#: from hold a bounded window and drop what falls out of it.  OmniMMI's proactive-alignment
#: harness is the reference -- its caller advances ``question_time`` a second at a time and each
#: call is served at most the last 30 one-second segments -- and AURA serves the same last 30
#: seconds.  Matching that number is what makes a long row's score a measurement of the model
#: rather than of how much context its server was configured with.
#:
#: This is a context window and not a scoring window.  Which seconds count as a hit is per row
#: and comes from the annotation; how far back a model may look is one number for the whole
#: benchmark, held equal across models for the same reason the pixel budget is (see
#: ``models.base.Dialect``: which turns exist is the benchmark's, only their wording is the
#: model's).  Nothing about a row changes when it is elided -- the targets, the reference
#: behaviour and the window it is scored against are all untouched.
#:
#: What survives the cut is the opening query, its ack, and any *spoken* assistant turn from the
#: elided seconds; the silent ones go with their chunks.  A live assistant does not forget what
#: it said, and the two-sided alert subtasks are built on exactly that: such a row asks for
#: the next alert after some correct ones, and dropping those would delete the evidence the
#: row turns on.  ``respond_after_long_silence`` -- the subtask this bound exists
#: for, its rows 66 to 231 seconds long -- has a verified-silent prefix, so there is never a
#: spoken turn in its elided region and the cut costs it nothing.
#:
#: Changing this number changes what every row longer than it is measured on, which is why it is
#: recorded in each run's ``run.json`` and why results either side of a change are not
#: comparable.
CONTEXT_WINDOW_CHUNKS = 30


class ConversationError(ValueError):
    """A prompt cannot be composed as the dialect describes."""


@dataclass
class Turn:
    """A chat turn whose content is either text or a flat list of text and media refs."""

    role: str
    content: Union[str, List[object]]


@dataclass
class Request:
    """Everything a model adapter needs for one completion."""

    turns: List[Turn]
    sample_id: str
    use_audio: bool
    media: MediaPlan = field(default_factory=MediaPlan)


def _seconds(value: float) -> str:
    """One decimal, the way every dialect that prints a time prints it."""
    return f"{float(value):.1f}"


def _compose(
    template: str, refs: Sequence[MediaRef], **fields: str
) -> Union[str, List[object]]:
    """Fill a dialect template, putting the media where its ``{media}`` slot is.

    Returns plain text when the turn carries no media, and a flat list otherwise: the order
    of that list is the order the model sees the frames in, which is why the media is spliced
    into the template rather than appended after it.
    """
    head, slot, tail = template.partition(MEDIA_SLOT)
    if not slot:
        if refs:
            raise ConversationError(
                f"this turn carries {len(refs)} media file(s) but the dialect's template has "
                f"no {MEDIA_SLOT} slot to put them in: {template!r}"
            )
        return template.format(**fields)
    if MEDIA_SLOT in tail:
        # Only the first slot is filled, so a second one would reach ``format`` as an unknown
        # field and surface as a bare KeyError several frames away from the template.
        raise ConversationError(
            f"this dialect's template has more than one {MEDIA_SLOT} slot, and a chunk's media "
            f"goes in one place: {template!r}"
        )
    parts: List[object] = []
    if head:
        parts.append(head.format(**fields))
    parts.extend(refs)
    if tail:
        parts.append(tail.format(**fields))
    return parts


def _layout(dialect: object, attribute: str) -> str:
    """One of the dialect's templates. Required, with no fallback on purpose.

    This module used to hold the corpus's own wording and lend it to any dialect that left a
    field empty.  That made a missing declaration invisible: a new baseline would be prompted
    in another model's words and score whatever that produced.  Every dialect now states its
    own, so the failure is this exception during the first poll instead.
    """
    value = getattr(dialect, attribute, "")
    if not value:
        raise ConversationError(
            f"this model's dialect declares no {attribute}: every dialect states its own, "
            "copied from its own repository -- see Dialect in models/base.py"
        )
    return str(value)


def _speech(dialect: object, text: str) -> str:
    """A spoken assistant turn, prefixed the way this model's repo prefixes one."""
    return dialect.as_speech(text)


def _reference(dialect: object, turn: AnnotationTurn) -> str:
    """One reference assistant turn, in this model's own words.

    The annotation's channel decides, never the text: it is kept apart from the wording
    precisely so that nothing downstream has to read an absent string as "stayed silent".
    Left untranslated, the prefix would show a baseline dozens of turns of a foreign silence
    token and then ask it to continue the pattern.
    """
    if not turn.spoke:
        return dialect.as_silence()
    return _speech(dialect, turn.text or "")


def chunk_turn(
    annotation: Annotation,
    chunk: Chunk,
    media: ChunkMedia,
    capabilities: Capabilities,
    use_audio: bool,
    plan: MediaPlan,
    dialect: object,
) -> Turn:
    """One streamed second as a user turn.

    The time carried is the *stream* clock, not the position and not the source second: the
    second an assistant spends speaking is never shown to it, so a row's presented clock can
    skip and the k-th chunk turn stops being the k-th second.
    """
    refs = media.refs(annotation.video, chunk, capabilities, use_audio, plan)
    content = _compose(
        _layout(dialect, "chunk_turn"),
        refs,
        start=_seconds(chunk.stream_start),
        end=_seconds(chunk.stream_end),
    )
    return Turn("user", content)


def _carried_speech(
    annotation: Annotation,
    chunk_idx: int,
    history: Optional[Mapping[int, str]],
    dialect: object,
) -> Optional[str]:
    """The assistant turn for an elided second, if it spoke; ``None`` if it stayed silent.

    Read through the same two sources as the in-window turns and in the same order, so that a
    second's history does not change meaning by falling out of the window: a replayed reply of
    the model's own is judged by this model's speak rule, and a reference turn by the channel
    the annotation recorded rather than by its text.
    """
    own = history.get(chunk_idx) if history else None
    if own is not None:
        return own if dialect.spoke(own) else None
    reference = annotation.reference_turn(chunk_idx)
    return _reference(dialect, reference) if reference.spoke else None


def build_request(
    annotation: Annotation,
    chunk_idx: int,
    media: ChunkMedia,
    capabilities: Capabilities,
    use_audio: bool,
    dialect: object,
    history: Optional[Mapping[int, str]] = None,
    context_window: int = CONTEXT_WINDOW_CHUNKS,
) -> Request:
    """Compose the conversation that ends with the chunk turn at ``chunk_idx``.

    Every assistant turn in the prefix is the annotation's own reference behaviour, translated
    into ``dialect``.  The model's own earlier answers are deliberately *not* fed back: each
    polled turn is one measurement taken against the dataset's history, so a miss on an earlier
    turn cannot change the question a later one is asked.  For the repeated-response profiles
    that history legitimately carries real alert text -- the model is being asked for the next
    alert after some correct ones -- and must not be flattened to silence.

    ``history`` overrides that prefix for the chunks it names, and exists for one mode.  A
    running-count row's reference turns are literally the answer -- "That is 3 times" on the
    third completion -- so replaying them would tell the model the number it is being asked to
    produce.  Such a row is instead run as one serial pass and each turn is answered against
    what the model itself said on the turns before it; the strings here are those replies and
    are inserted verbatim rather than through ``_reference``, because they are already in this
    model's own words and re-prefixing them would double the dialect's speech marker.

    The query turn is the row's ``query.text`` as published, which since schema v2 already
    carries the silence clause its subtask is scored against; nothing is appended here.  The ack
    that follows is the row's own, left as the file writes it: answering the clause in the
    assistant's voice was measured and cost pass rate on both halves of the scored pair, so the
    request states the rule once, on the turn that asks for it.

    ``context_window`` seconds of stream reach the prompt at most, counted back from
    ``chunk_idx``; the seconds before that are dropped along with their silent turns, and what
    they spoke is carried over as bare assistant turns after the ack -- see
    :data:`CONTEXT_WINDOW_CHUNKS`.  0 replays the whole row, which is what this function did
    before the bound existed and is kept for reproducing a run made without it.

    ``dialect`` is loosely typed because ``models.base`` imports this module.
    """
    plan = MediaPlan()
    turns: List[Turn] = [
        Turn("system", _layout(dialect, "system_prompt")),
        Turn("user", _compose(
            _layout(dialect, "query_turn"),
            (),
            text=annotation.query_text,
            start=_seconds(annotation.query_time),
        )),
        Turn("assistant", _speech(dialect, annotation.ack)),
    ]
    prefix = annotation.prefix(chunk_idx)
    if context_window > 0 and len(prefix) > context_window:
        elided, prefix = prefix[:-context_window], prefix[-context_window:]
        for chunk in elided:
            carried = _carried_speech(annotation, chunk.chunk_idx, history, dialect)
            if carried is not None:
                turns.append(Turn("assistant", carried))
    for chunk in prefix:
        turns.append(
            chunk_turn(annotation, chunk, media, capabilities, use_audio, plan, dialect)
        )
        if chunk.chunk_idx == chunk_idx:
            # The turn under test: the model answers here, so no assistant turn follows.
            break
        own = history.get(chunk.chunk_idx) if history else None
        if own is not None:
            turns.append(Turn("assistant", own))
            continue
        reference = annotation.reference_turn(chunk.chunk_idx)
        turns.append(Turn("assistant", _reference(dialect, reference)))
    return Request(
        turns=turns, sample_id=annotation.sample_id, use_audio=use_audio, media=plan
    )


def turn_text(turn: Turn) -> str:
    """Flatten a turn's content to text, dropping media refs. Used by text-only judges."""
    if isinstance(turn.content, str):
        return turn.content
    return "".join(piece for piece in turn.content if isinstance(piece, str))
