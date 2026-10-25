"""The published annotation: what the benchmark actually is.

A row here describes a proactive-response decision problem and nothing else -- which video,
which second of it each stream chunk shows, what the user asked for, which turns are scored
and what the reference behaviour on each turn is.  There is no message list, no frame path
and no audio path, because those are all *one model's* rendering of the problem:
LiveChatOmni's ``<chunk><video>`` turn, AURA's ``<|silent|>``, ROMA's two frames per second.
Publishing any of them as the data would make the set that model's set.  Each adapter
assembles its own prompt from these fields (:mod:`conversation`) and derives its own pixels
and samples from the timestamps (:mod:`extraction`).

Two fields are easy to confuse and mixing them up silently scores the wrong second:

``chunk_idx``
    Position on the *stream clock*, in seconds from the start of this row's stream.  This is
    the number ``targets`` and the response window speak in.  It is not contiguous: a second
    the reference assistant spoke over is dropped from the stream, so a row may run
    ``0..9, 11, 12`` with no chunk 10.

``position``
    Index into ``stream``, i.e. how many chunk turns precede this one.  This is what the
    conversation prefix is built from.

:meth:`Annotation.chunk` maps the first onto the second and raises when a ``chunk_idx`` is
not in the stream, so the confusion cannot pass silently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .dataset import load_rows, plain
from .suite import (
    LAST_TURN,
    RUNNING_COUNT,
    SUBTASKS,
    TWO_SIDED_ALERTS,
    WINDOW,
    metric_name,
)

SCHEMA = "proactive_bench_annotation_v2"

#: The keys a published ``annotation`` block keeps, and the only ones anything reads: ``judge``
#: takes ``instruction`` as the request a reply is checked against, and the rest are the fallback
#: chain it derives the awaited occurrence from.  The delivery's own block carried 72 distinct
#: keys across the set -- rationales, audit verdicts, source windows -- which no consumer opened
#: and which made the set three times the size it needed to be.  Named here, beside the reader,
#: so a rebuild keeps what is read rather than what happened to be in the delivery.
PUBLISHED_META = (
    "instruction",
    "response_events",
    "first_evidence_description",
    "activity_description",
    "event_summary",
)
#: Likewise within each entry of ``response_events``.
PUBLISHED_EVENT_FIELDS = (
    "evidence_chunk",
    "event_description",
    "first_evidence_description",
)

SILENCE_CHANNEL = "silence"
RESPOND_CHANNEL = "respond"

SPEAK = "speak"
SILENCE = "silence"


class AnnotationError(ValueError):
    """A row does not describe a scorable problem."""


@dataclass(frozen=True)
class Chunk:
    """One second of stream, and the second of source video it shows."""

    chunk_idx: int
    position: int
    stream_start: float
    stream_end: float
    source_chunk: int
    source_start: float
    source_end: float

    @property
    def seconds(self) -> float:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class Turn:
    """What the reference assistant did on one chunk turn.

    ``text`` is the reference alert when it spoke and ``None`` when it stayed silent, kept
    apart from the channel so that no adapter has to guess whether an empty string means
    "silent" or "spoke, wording unrecorded".
    """

    chunk_idx: int
    channel: str
    text: Optional[str] = None

    @property
    def spoke(self) -> bool:
        return self.channel == RESPOND_CHANNEL


@dataclass(frozen=True)
class Target:
    """One scored turn: what is expected of the model on this chunk."""

    chunk_idx: int
    score_as: str
    expected: Optional[str] = None
    #: Why this turn is not scored.  Every excluded turn in the set is excluded for the same
    #: reason, which the manifest states once as ``exclusion_reason``; the rows no longer repeat
    #: it, so this is empty unless a set is read that still carries it.
    excluded_reason: str = ""
    #: Which alert this turn is one of the acceptable seconds for, in a ``two_sided_alerts`` row.
    #: Several turns share one index and speaking on any of them catches that alert once, so the
    #: index -- not the turn -- is the unit the row is scored on.  ``None`` everywhere else,
    #: including on the silence turns of a two-sided row and on every turn of every other mode,
    #: where each speak turn is its own decision and grouping would silently forgive a miss.
    alert_index: Optional[int] = None

    @property
    def wants_speech(self) -> bool:
        return self.score_as == SPEAK


@dataclass(frozen=True)
class Video:
    dataset: str
    video_id: str
    relpath: str
    chunk_seconds: float = 1.0


@dataclass(frozen=True)
class Annotation:
    """One benchmark row."""

    sample_id: str
    plan_id: str
    subtask: str
    mode: str
    metric: str
    video: Video
    query_text: str
    query_time: float
    ack: str
    stream: Tuple[Chunk, ...]
    reference: Tuple[Turn, ...]
    targets: Tuple[Target, ...]
    excluded: Tuple[Target, ...] = ()
    response_window: Optional[Tuple[int, int]] = None
    #: How many times the counted unit completes in this row's stream. The ground truth of the
    #: ``running_count`` mode, and the only one it has: the row is scored on the total, not on
    #: which second each completion was announced at.
    total_reps: Optional[int] = None
    #: The wording the row's query asks a count to be reported in, as a regular expression with
    #: the number as its one group. Carried per row rather than fixed in the scorer because it
    #: is part of what the query asked for, and it is used to judge the *wording* only -- the
    #: count itself is read more leniently, so a model is never failed for punctuation.
    reply_pattern: str = ""
    meta: Dict[str, object] = field(default_factory=dict)
    _by_chunk: Dict[int, Chunk] = field(default_factory=dict, repr=False)
    _reference_by_chunk: Dict[int, Turn] = field(default_factory=dict, repr=False)

    # -- the query as asked ----------------------------------------------------------

    # ``query_text`` is the query as a model is asked it, silence clause included: the clause is
    # part of what the row requests, so it is published in the row rather than appended by
    # whoever builds the request.  A set whose files did not carry it could be read as asking
    # only to be told when something happens, while the scorer fails any speech before that --
    # which is the mismatch the clause was added to remove.  ``suite.hold_clause`` remains the
    # authority a rebuild copies from; the set's build step is what put it here.

    # -- lookups ---------------------------------------------------------------------

    def chunk(self, chunk_idx: int) -> Chunk:
        try:
            return self._by_chunk[int(chunk_idx)]
        except KeyError:
            raise AnnotationError(
                f"{self.sample_id}: chunk_idx {chunk_idx} is not in the stream "
                f"({self.stream[0].chunk_idx}..{self.stream[-1].chunk_idx}); a stream-clock "
                f"index was probably confused with a list position"
            ) from None

    def position_of(self, chunk_idx: int) -> int:
        return self.chunk(chunk_idx).position

    def prefix(self, chunk_idx: int) -> Tuple[Chunk, ...]:
        """Every chunk up to and including ``chunk_idx``, in stream order.

        This is the history a poll at ``chunk_idx`` is made against.
        """
        return self.stream[: self.position_of(chunk_idx) + 1]

    def reference_turn(self, chunk_idx: int) -> Turn:
        turn = self._reference_by_chunk.get(int(chunk_idx))
        if turn is None:
            raise AnnotationError(
                f"{self.sample_id}: no reference turn for chunk_idx {chunk_idx}"
            )
        return turn

    @property
    def first_chunk_idx(self) -> int:
        return self.stream[0].chunk_idx

    @property
    def last_chunk_idx(self) -> int:
        return self.stream[-1].chunk_idx

    @property
    def scored_chunks(self) -> Tuple[int, ...]:
        return tuple(target.chunk_idx for target in self.targets)

    @property
    def alert_groups(self) -> Tuple[Tuple[Target, ...], ...]:
        """The speak turns of a ``two_sided_alerts`` row, grouped per alert and in stream order.

        One tuple per alert, each holding the seconds on which speaking counts as catching it:
        a single second for the strict subtask, the three granted ones for the grace subtask.
        Empty for every other mode, which groups nothing.
        """
        grouped: Dict[int, List[Target]] = {}
        for target in self.targets:
            if target.alert_index is None:
                continue
            grouped.setdefault(int(target.alert_index), []).append(target)
        return tuple(
            tuple(sorted(items, key=lambda item: item.chunk_idx))
            for _, items in sorted(grouped.items())
        )

    @property
    def silence_targets(self) -> Tuple[Target, ...]:
        """The scored turns whose reference behaviour is silence, in stream order."""
        return tuple(
            sorted(
                (target for target in self.targets if not target.wants_speech),
                key=lambda item: item.chunk_idx,
            )
        )

    def source_path(self, root: str) -> str:
        return os.path.join(root, "videos", self.video.relpath)


def _chunk(entry: dict, expected_position: int, sample_id: str) -> Chunk:
    chunk = Chunk(
        chunk_idx=int(entry["chunk_idx"]),
        position=int(entry["position"]),
        stream_start=float(entry["stream_start"]),
        stream_end=float(entry["stream_end"]),
        source_chunk=int(entry["source_chunk"]),
        source_start=float(entry["source_start"]),
        source_end=float(entry["source_end"]),
    )
    if chunk.position != expected_position:
        raise AnnotationError(
            f"{sample_id}: stream entry {expected_position} claims position "
            f"{chunk.position}; the list order is the prefix order and cannot disagree"
        )
    return chunk


def _target(entry: dict) -> Target:
    index = entry.get("alert_index")
    return Target(
        chunk_idx=int(entry["chunk_idx"]),
        score_as=str(entry["score_as"]),
        expected=entry.get("expected"),
        excluded_reason=str(entry.get("excluded_reason") or ""),
        alert_index=int(index) if index is not None else None,
    )


def _check_two_sided(targets: Sequence[Target], sample_id: str) -> None:
    """That a two-sided row really carries both sides, and that its alerts are well formed.

    The mode's whole reason to exist is that one row supplies both rates of its paired score, so
    a row missing either side is not a weaker row but an unscorable one: its geometric mean would
    silently fall back to the side that is present and read as a number measured on both.

    The rest is about the grouping.  ``alert_index`` decides how much speech one alert is worth,
    so a malformed group is not a data blemish -- a group that lost a second narrows the grace it
    is meant to grant, an index that repeats a chunk scores one second twice, and a chunk that is
    both an alert second and a silence target asks for speech and quiet at once.  All three would
    still score, which is why they are rejected here rather than left to be noticed in a table.
    """
    speak = [target for target in targets if target.wants_speech]
    quiet = [target for target in targets if not target.wants_speech]
    if not speak or not quiet:
        raise AnnotationError(
            f"{sample_id}: scored in {TWO_SIDED_ALERTS} mode with "
            f"{len(speak)} speak and {len(quiet)} silence target(s); the mode pairs the two"
            " rates of one row, so both sides have to be on it"
        )
    chunks = [target.chunk_idx for target in targets]
    if len(set(chunks)) != len(chunks):
        repeated = sorted({idx for idx in chunks if chunks.count(idx) > 1})
        raise AnnotationError(
            f"{sample_id}: chunk_idx {repeated[0]} is scored by more than one target"
        )
    unindexed = [target.chunk_idx for target in speak if target.alert_index is None]
    if unindexed:
        raise AnnotationError(
            f"{sample_id}: speak target at chunk {unindexed[0]} carries no alert_index; in "
            f"{TWO_SIDED_ALERTS} mode the alert is the scored unit, not the turn"
        )
    misindexed = [target.chunk_idx for target in quiet if target.alert_index is not None]
    if misindexed:
        raise AnnotationError(
            f"{sample_id}: silence target at chunk {misindexed[0]} carries an alert_index; "
            "only the seconds an alert may be spoken on belong to a group"
        )
    grouped: Dict[int, List[int]] = {}
    for target in speak:
        grouped.setdefault(int(target.alert_index), []).append(target.chunk_idx)
    indices = sorted(grouped)
    if indices != list(range(len(indices))):
        raise AnnotationError(
            f"{sample_id}: alert_index values are {indices}; they number this row's alerts "
            "and so must run 0..n-1 without a gap"
        )
    groups = [sorted(grouped[index]) for index in indices]
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise AnnotationError(
            f"{sample_id}: alert groups span {sorted(widths)} seconds; one row grants the same "
            "delay at every alert, so a differing width means a group lost a second"
        )
    for index, group in zip(indices, groups):
        holes = [
            (first, second)
            for first, second in zip(group, group[1:])
            if second != first + 1
        ]
        if holes:
            raise AnnotationError(
                f"{sample_id}: alert {index} skips from chunk {holes[0][0]} to {holes[0][1]}; "
                "the seconds an alert may be spoken on are consecutive by construction"
            )
    for earlier, later in zip(groups, groups[1:]):
        if earlier[-1] >= later[0]:
            raise AnnotationError(
                f"{sample_id}: alert groups overlap or run out of order at chunk {later[0]}"
            )


def from_row(row: dict) -> Annotation:
    """Build an :class:`Annotation` from one annotation-set row.

    Every field the harness later relies on is checked here rather than where it is used:
    a row that has drifted from the schema should fail while it is still one identifiable
    row, not as a puzzling media or scoring error thousands of polls later.
    """
    row = plain(row)
    schema = str(row.get("schema") or "")
    if schema != SCHEMA:
        raise AnnotationError(
            f"row schema is {schema!r}, expected {SCHEMA!r}; point --layout at the set this "
            f"file belongs to"
        )
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        raise AnnotationError("row has no sample_id")

    stream_entries = row.get("stream") or []
    if not stream_entries:
        raise AnnotationError(f"{sample_id}: no stream chunks")
    stream = tuple(
        _chunk(entry, index, sample_id) for index, entry in enumerate(stream_entries)
    )
    by_chunk = {chunk.chunk_idx: chunk for chunk in stream}
    if len(by_chunk) != len(stream):
        raise AnnotationError(f"{sample_id}: the stream repeats a chunk_idx")

    reference = tuple(
        Turn(
            chunk_idx=int(entry["chunk_idx"]),
            channel=str(entry.get("channel") or SILENCE_CHANNEL),
            text=entry.get("text"),
        )
        for entry in row.get("reference_turns") or []
    )
    reference_by_chunk = {turn.chunk_idx: turn for turn in reference}
    missing = [chunk.chunk_idx for chunk in stream if chunk.chunk_idx not in reference_by_chunk]
    if missing:
        # Without a reference turn per chunk the prefix cannot be replayed, and a poll would
        # silently be made against a shorter history than the annotation describes.
        raise AnnotationError(
            f"{sample_id}: {len(missing)} chunk(s) have no reference turn, first {missing[0]}"
        )

    targets = tuple(_target(entry) for entry in row.get("targets") or [])
    if not targets:
        raise AnnotationError(f"{sample_id}: no scored targets")
    unknown = [target.chunk_idx for target in targets if target.chunk_idx not in by_chunk]
    if unknown:
        raise AnnotationError(
            f"{sample_id}: target chunk_idx {unknown[0]} is not in the stream"
        )
    excluded = tuple(_target(entry) for entry in row.get("excluded_turns") or [])

    subtask = str(row.get("subtask") or "")
    scoring = row.get("scoring") or {}
    mode = str(scoring.get("mode") or "")
    if not mode:
        item = SUBTASKS.get(subtask)
        mode = item.mode if item else LAST_TURN
    window_pair: Optional[Tuple[int, int]] = None
    window = scoring.get("response_window")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        window_pair = (int(window[0]), int(window[1]))
    if mode == WINDOW and window_pair is None:
        raise AnnotationError(
            f"{sample_id}: scored in {WINDOW} mode but carries no response_window"
        )
    total_reps = scoring.get("total_reps")
    total_reps = int(total_reps) if total_reps is not None else None
    if mode == RUNNING_COUNT and total_reps is None:
        # The mode's entire pass rule is "the last count equals this number", so a row without
        # it is not scorable -- and would otherwise pass or fail on a comparison against None.
        raise AnnotationError(
            f"{sample_id}: scored in {RUNNING_COUNT} mode but carries no total_reps"
        )
    if mode == TWO_SIDED_ALERTS:
        _check_two_sided(targets, sample_id)

    video_entry = row.get("video") or {}
    video = Video(
        dataset=str(video_entry.get("dataset") or ""),
        video_id=str(video_entry.get("video_id") or row.get("video_id") or ""),
        relpath=str(video_entry.get("relpath") or ""),
        chunk_seconds=float(video_entry.get("chunk_seconds") or 1.0),
    )
    if not video.relpath:
        raise AnnotationError(f"{sample_id}: no video relpath to derive media from")

    query = row.get("query") or {}
    return Annotation(
        sample_id=sample_id,
        plan_id=str(row.get("plan_id") or ""),
        subtask=subtask,
        mode=mode,
        metric=str(scoring.get("metric") or metric_name(subtask)),
        video=video,
        query_text=str(query.get("text") or ""),
        query_time=float(query.get("stream_time") or 0.0),
        ack=str(row.get("ack") or ""),
        stream=stream,
        reference=reference,
        targets=targets,
        excluded=excluded,
        response_window=window_pair,
        total_reps=total_reps,
        reply_pattern=str(scoring.get("reply_pattern") or ""),
        meta=dict(row.get("annotation") or {}),
        _by_chunk=by_chunk,
        _reference_by_chunk=reference_by_chunk,
    )


def load(path: str, limit: int = 0) -> List[Annotation]:
    return [from_row(row) for row in load_rows(path, limit=limit)]


def iter_rows(paths: Sequence[str], limit: int = 0) -> Iterator[Annotation]:
    for path in paths:
        for row in load_rows(path, limit=limit):
            yield from_row(row)
