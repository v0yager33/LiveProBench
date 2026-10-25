"""The scoring modes.

``pass`` is a pure speak/silence rule in the first-event, repeated and window modes so that
those sets stay comparable with each other and with OmniMMI PA.  Content correctness is
reported separately as ``content_ok``/``pass_strict`` and never folded into ``pass``.

``running_count`` is the one mode whose ``pass`` is not a speak/silence rule, because speaking
is not what it measures: the row asks for a count and is scored on the number the model ended
up at.  It keeps the same split -- the count is read leniently for ``pass``, and whether the
model used the wording the query asked for is reported beside it as ``content_ok``.  It is also
the one mode whose turns are not independent; see :func:`Evaluator._evaluate_running_count`.

``two_sided_alerts`` is the one mode that asks a row both questions -- speak at each alert, stay
quiet between them -- so that the paired score in :func:`metrics.subsets` is two rates of one
behaviour rather than one rate from each of two videos.

Turns are addressed by ``chunk_idx`` -- the stream clock -- and never by their position in a
message list.  The two diverge: a second the reference assistant spoke over is dropped from
the stream, so a row can run ``0..9, 11`` and the eleventh chunk turn is second 11.  The
annotation is the only thing that knows which is which, so scoring asks it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

from . import SILENCE
from .annotation import SPEAK, Annotation, Target
from .conversation import CONTEXT_WINDOW_CHUNKS, build_request
from .media import ChunkMedia
from .models.base import ProactiveModel
from .suite import (
    LAST_TURN,
    RUNNING_COUNT,
    SERIAL,
    TWO_SIDED_ALERTS,
    WINDOW,
)

#: The wording every counting query asks a count to be reported in, used for a row that names
#: no pattern of its own.  Matched case-insensitively and by search, so trailing punctuation or
#: a capitalised variant is still the asked-for wording.
DEFAULT_COUNT_PATTERN = r"That is (\d+) times"

_INTEGER = re.compile(r"\d+")


def model_spoke(text: Optional[str]) -> bool:
    """The speak/silence rule for the dialect the corpus was annotated in -- our model's.

    Deliberately literal: an empty completion and the bare silence token are silence, and
    anything else is speech.  Loosening this -- for instance by also treating "nothing has
    happened yet" as silence -- would make the numbers incomparable with the reference runs
    and with OmniMMI PA.

    A baseline is *not* judged by this function.  It names silence differently (AURA
    ``<|silent|>``, JoyAI ``</silence>``) and MOSS-VL answers one poll with a run of silence
    tokens, so each model is judged by its own ``dialect.spoke``; this stays as the rule for
    reading the annotation's own reference turns.
    """
    if text is None:
        return False
    stripped = str(text).strip()
    return not (stripped == "" or stripped == SILENCE)


def row_pass(mode: str, predictions: List[dict]) -> bool:
    """Apply the mode's rule to turns that have already been judged.

    Dispatch is on ``scoring.mode``, never on the subtask name, so a new subtask that reuses a
    mode needs no new scoring code.  Reads each turn's recorded ``spoke`` rather than
    re-deriving it from the text, so the rule stays independent of which model's dialect
    produced the transcript and a saved run can be rescored without knowing it.

    ``running_count`` is absent on purpose: its rule needs the row's ground-truth total, which
    is not in the turns, so it is applied by :func:`count_breakdown` instead.
    """
    if not predictions:
        return False
    if mode == LAST_TURN:
        return bool(predictions[-1].get("spoke"))
    if mode == SERIAL:
        return all(not item.get("spoke") for item in predictions)
    if mode == TWO_SIDED_ALERTS:
        # Both sides of the row, from the turns alone: a turn belonging to an alert is grouped by
        # its ``alert_index`` and one of the group speaking catches that alert, while a turn with
        # no index is one the row asks quiet on.  A group polled only until it was caught is still
        # represented by the turns that were polled, so no group can go missing from this.
        caught: Dict[int, bool] = {}
        for item in predictions:
            index = item.get("alert_index")
            if index is None:
                if item.get("spoke"):
                    return False
                continue
            caught[int(index)] = caught.get(int(index), False) or bool(item.get("spoke"))
        return bool(caught) and all(caught.values())
    return False


def response_breakdown(predictions: List[dict]) -> Dict[str, Any]:
    """The three questions a repeated-response row answers, rather than one.

    Such a row holds three to six reference alerts, and collapsing them into a single boolean
    loses the distinction the set exists to draw: catching the first event is a different skill
    from still catching the fourth.  ``all_responses`` is the headline of the three -- but it is
    not the row's ``pass`` in ``two_sided_alerts``, where the row also has to have stayed quiet
    between the alerts; the other two say how a failing row failed.

    ``first_response`` is not "spoke at least once" -- a model that misses the opening alert and
    catches the later ones fails this one and passes ``two_or_more_responses``.

    One item per *alert*, which in ``two_sided_alerts`` is one per group of polled turns; the
    caller passes the per-alert verdicts rather than the per-turn ones, so the rates keep a
    denominator of alerts however many seconds an alert was granted.
    """
    spoke = [bool(item.get("spoke")) for item in predictions]
    return {
        "response_turns": len(spoke),
        "responses_spoken": sum(spoke),
        "first_response": bool(spoke and spoke[0]),
        "two_or_more_responses": sum(spoke) >= 2,
        "all_responses": bool(spoke) and all(spoke),
    }


def count_pattern(source: str) -> Pattern[str]:
    """Compile a row's asked-for reply wording, defaulting to the corpus's own.

    Checked for its one capturing group here rather than at read time: a pattern without it
    would otherwise fail on the first row that happens to match, hundreds of polls in.
    """
    text = source or DEFAULT_COUNT_PATTERN
    try:
        pattern = re.compile(text, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"reply_pattern {text!r} is not a regular expression: {exc}") from None
    if pattern.groups < 1:
        raise ValueError(
            f"reply_pattern {text!r} has no capturing group; the count is read from group 1"
        )
    return pattern


def read_count(text: str, pattern: Pattern[str], dialect: Any) -> Tuple[Optional[int], bool]:
    """The number a reply reports, and whether it was reported in the asked-for wording.

    Two readings, because they answer different questions.  ``pass`` is about whether the model
    counted correctly, so the number is read leniently: the asked-for pattern first and, failing
    that, the last integer left in the reply -- a model that says "That's 4 reps" has counted,
    and failing it for that would turn the headline into a measure of instruction-following.
    Whether the wording *was* the asked-for one is returned beside it and reported as
    ``content_ok``, which is where the rest of the harness keeps phrasing.

    The pattern is tried before the integers, so "that is 4 times, 2 to go" reads as 4 and not
    as 2.

    The dialect's own markers come off first: MOSS-VL ends a speaking turn with its silence
    token and JoyAI prefixes one, and neither is something the model said.
    """
    body = str(text or "")
    for token in (dialect.silence, dialect.speak_prefix) + tuple(dialect.hidden or ()):
        if token:
            body = body.replace(token, " ")
    found = pattern.search(body)
    if found:
        return int(found.group(1)), True
    loose = _INTEGER.findall(body)
    return (int(loose[-1]) if loose else None), False


def count_breakdown(predictions: List[dict], total_reps: int) -> Dict[str, Any]:
    """What one running-count row's transcript says, and whether it passed.

    The rule reads the *last* count announced.  The row's ground truth is a total and nothing
    else -- which second each completion was announced at is not annotated for every source, and
    is not what the set claims to measure -- so the number the model ended the stream on is the
    answer.  Everything else here is diagnosis of how it got there: whether the counts ran
    ``1..N`` without repeating one, how many turns it spoke on at all, and how many of those
    named no number.

    A row that never announces a count fails, because ``final_count`` is then ``None`` and
    equals no total.  That is what makes the mode degeneracy-proof in both directions: silence
    throughout fails here, and announcing on every second overshoots the total and fails too.
    """
    announced = [
        (int(item["chunk_idx"]), int(item["count"]))
        for item in predictions
        if item.get("count") is not None
    ]
    values = [value for _, value in announced]
    final = values[-1] if values else None
    spoken = [item for item in predictions if item.get("spoke")]
    formats = [
        bool(item.get("count_format_ok"))
        for item in predictions
        if item.get("count") is not None
    ]
    target = int(total_reps)
    return {
        "count_target": target,
        "final_count": final,
        "final_count_chunk": announced[-1][0] if announced else None,
        "count_exact": final == target,
        "count_error": (final - target) if final is not None else None,
        # The tail, because the rule is about where the count ended up: these are the
        # announcements to have in front of you when a row is disputed.
        "count_tail": [{"chunk_idx": idx, "count": value} for idx, value in announced[-3:]],
        "counts_announced": values,
        "count_turns": len(values),
        "spoke_turns": len(spoken),
        # Spoke, but named no number at all: the turn broke the reply format badly enough that
        # there is nothing to read a count out of.
        "spoke_without_count": sum(1 for item in spoken if item.get("count") is None),
        # The strong form of "no count reported more than once": a clean transcript counts
        # 1, 2, ... N. Reported, never required -- a model can arrive at the right total having
        # skipped or re-announced a number on the way, and that is a different failure.
        "counts_sequential": values == list(range(1, len(values) + 1)),
        "repeated_counts": sorted({value for value in values if values.count(value) > 1}),
        "count_format_ok": all(formats) if formats else None,
    }


@dataclass
class EvalConfig:
    media: Optional[ChunkMedia] = None
    judge: Optional[Any] = None
    stop_on_first_failure: bool = False
    #: Seconds of stream each poll may look back over. Carried here rather than read from
    #: ``conversation`` at the call site so that a run's bound is one value chosen once, on
    #: record in ``run.json``, and the same for every row and every poll of the run.
    context_window: int = CONTEXT_WINDOW_CHUNKS


class Evaluator:
    """Scores one annotated row against one model."""

    def __init__(self, model: ProactiveModel, config: EvalConfig) -> None:
        self.model = model
        self.config = config
        if config.media is None:
            raise ValueError("scoring needs a ChunkMedia to resolve each chunk's frames")
        self.media: ChunkMedia = config.media

    @property
    def use_audio(self) -> bool:
        """Feed every chunk's audio track when the model can consume it.

        The benchmark has one stream-input definition. A model without an audio encoder --
        or a run started with ``--audio off`` -- keeps the frames and simply is not fed the
        track, and the media plan records which of the two it was.
        """
        return bool(self.model.capabilities.audio)

    def _ask(
        self, annotation: Annotation, chunk_idx: int, want_audio: bool,
        history: Optional[Dict[int, str]] = None,
    ) -> Tuple[str, Dict[str, Any], Any]:
        """Poll the turn that follows ``chunk_idx``. Returns ``(response, diagnostics, request)``.

        The prefix is the annotation's own, so the polls of one row are independent of each
        other and of the order they are made in.  ``history`` replaces it with the model's own
        replies for the chunks it names; only the running-count mode passes one, and only
        because that mode's reference turns are the answer -- see ``_evaluate_running_count``.

        The diagnostics channel exists for models whose speak/silence decision is not the
        text itself -- ROMA decides on a gate probability -- so that the number the decision
        actually turned on is recorded per turn and a threshold can be re-swept offline.
        """
        request = build_request(
            annotation,
            chunk_idx,
            self.media,
            self.model.capabilities,
            want_audio,
            dialect=self.model.dialect,
            history=history,
            context_window=self.config.context_window,
        )
        response, diagnostics = self.model.generate_with_diagnostics(request)
        return response, diagnostics, request

    def _targets(self, annotation: Annotation) -> List[Target]:
        """The scored turns, in stream order.

        Excluded turns are not here and are never polled for a score: the first chunk turn
        is dropped from every subtask because a model's own repository may open its stream
        with the query attached to the first frames, which makes that one poll a different
        question asked with a different history.  It is still replayed as history.
        """
        return sorted(annotation.targets, key=lambda target: target.chunk_idx)

    def _base_record(self, annotation: Annotation) -> Dict[str, Any]:
        return {
            "sample_id": annotation.sample_id,
            "plan_id": annotation.plan_id,
            "video_id": annotation.video.video_id,
            "subtask": annotation.subtask,
            "model": self.model.name,
            "mode": annotation.mode,
            "metric": annotation.metric,
            "status": "ok",
        }

    def _prediction(self, annotation: Annotation, target: Target, response: str,
                    spoke: bool) -> Dict[str, Any]:
        """The per-turn record, addressed both ways.

        ``chunk_idx`` is what the scoring rules speak; ``source_chunk`` is which second of
        the source video it was, which is also the name of the frame and wav files on disk,
        so a disputed turn can be looked at rather than argued about.
        """
        chunk = annotation.chunk(target.chunk_idx)
        record: Dict[str, Any] = {
            "chunk_idx": target.chunk_idx,
            "source_chunk": chunk.source_chunk,
            "expected": target.expected,
            "expected_action": target.score_as,
            "response": response,
            "spoke": spoke,
        }
        # Only where the row groups turns, so that a mode which scores each turn on its own does
        # not carry a key that would read as "this turn was one of several chances".
        if target.alert_index is not None:
            record["alert_index"] = int(target.alert_index)
        return record

    def evaluate(self, annotation: Annotation) -> Dict[str, Any]:
        if annotation.mode == RUNNING_COUNT:
            if annotation.total_reps is None:
                raise ValueError(
                    f"{annotation.sample_id} is scored in {RUNNING_COUNT} mode but carries "
                    "no total_reps"
                )
            return self._evaluate_running_count(annotation, annotation.total_reps)
        if annotation.mode == WINDOW:
            if annotation.response_window is None:
                raise ValueError(
                    f"{annotation.sample_id} is scored in {WINDOW} mode but carries no window"
                )
            return self._evaluate_window(annotation, annotation.response_window)
        if annotation.mode == TWO_SIDED_ALERTS:
            return self._evaluate_two_sided(annotation)
        return self._evaluate_turns(annotation)

    def _evaluate_turns(self, annotation: Annotation) -> Dict[str, Any]:
        """The three turn-by-turn modes share one polling loop.

        All of them walk the target turns in order against the annotation's own history; they
        differ only in the final pass rule, so the loop is written once.
        """
        want_audio = self.use_audio
        predictions: List[dict] = []
        media_plan = None
        started = time.time()
        for target in self._targets(annotation):
            response, diagnostics, request = self._ask(
                annotation, target.chunk_idx, want_audio
            )
            media_plan = request.media
            spoke = self.model.dialect.spoke(response)
            prediction = self._prediction(annotation, target, response, spoke)
            prediction["turn_pass"] = spoke if target.wants_speech else not spoke
            # Nested, never merged: a backend must not be able to shadow a scored field.
            if diagnostics:
                prediction["diagnostics"] = diagnostics
            # Only a speak target has content to be right or wrong about; a silence target
            # is fully decided by the token itself.
            if self.config.judge and target.wants_speech and spoke:
                prediction.update(self.config.judge.judge(annotation, target, response))
            predictions.append(prediction)
            if self.config.stop_on_first_failure and not prediction["turn_pass"]:
                break

        judged = [item for item in predictions if item.get("content_ok") is not None]
        content_ok = all(bool(item["content_ok"]) for item in judged) if judged else None
        passed = row_pass(annotation.mode, predictions)
        record = self._base_record(annotation)
        record.update({
            "pass": passed,
            "content_ok": content_ok,
            "pass_strict": bool(passed and content_ok is not False),
            "media": media_plan.as_dict() if media_plan else {},
            "predictions": predictions,
            "latency_sec": round(time.time() - started, 3),
        })
        return record

    def _evaluate_window(
        self, annotation: Annotation, window: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Poll the row's chunk turns in order and score where the model first speaks.

        The fuzzy request has no per-chunk label, so the run stops at the first non-silent
        turn and that turn is the entire prediction: inside the window passes, before it is
        ``early``, never speaking by the window's last chunk is ``missed``.
        """
        start, end = window
        want_audio = self.use_audio
        predictions: List[dict] = []
        first_target: Optional[Target] = None
        first_chunk: Optional[int] = None
        media_plan = None
        started = time.time()
        for target in self._targets(annotation):
            if target.chunk_idx > end:
                break
            response, diagnostics, request = self._ask(
                annotation, target.chunk_idx, want_audio
            )
            media_plan = request.media
            spoke = self.model.dialect.spoke(response)
            prediction = self._prediction(annotation, target, response, spoke)
            prediction["in_window"] = start <= target.chunk_idx <= end
            if diagnostics:
                prediction["diagnostics"] = diagnostics
            predictions.append(prediction)
            if spoke:
                first_target = target
                first_chunk = target.chunk_idx
                break

        passed = first_chunk is not None and start <= first_chunk <= end
        verdict = "in_window" if passed else ("early" if first_chunk is not None else "missed")
        record = self._base_record(annotation)
        record.update({
            "pass": passed,
            "verdict": verdict,
            "response_window": [start, end],
            "first_response_chunk": first_chunk,
            # How far past the window's onset the model needed; the window is exactly the
            # budget for that recognition delay, so this says how much of it was used.
            "response_delay_chunks": (first_chunk - start) if passed else None,
            "content_ok": None,
            "media": media_plan.as_dict() if media_plan else {},
            "predictions": predictions,
            "latency_sec": round(time.time() - started, 3),
        })
        # Only an in-window alert has content worth judging: an early one is by construction
        # a report of something that has not happened, and its wording changes nothing.
        if self.config.judge and passed and first_target is not None:
            verdict_fields = self.config.judge.judge(
                annotation, first_target, predictions[-1]["response"]
            )
            record.update(verdict_fields)
            predictions[-1].update(verdict_fields)
        record["pass_strict"] = bool(passed and record.get("content_ok") is not False)
        return record

    def _evaluate_two_sided(self, annotation: Annotation) -> Dict[str, Any]:
        """Score one row's speaking and its silence, so the paired score is of one behaviour.

        Every other mode asks a row one of the two questions, which leaves ``metrics.subsets``
        pairing a rate measured on one video against a rate measured on another.  Here the alerts
        and the quiet stretches are both this row's, so the geometric mean the subset reports is
        finally about one model on one recording.

        An alert is a *group* of seconds and is caught by speech on any of them, so the group is
        polled a second at a time and stops at the first one the model speaks on.  Each of those
        polls replaces the reference's own alert with the model's silence for the seconds already
        passed: the reference alerts on the group's first second -- that is what makes it the
        alert -- and replaying it would tell a model polled a second later that the alert has
        already been given, so the grace this subtask grants would be a poll no model has a reason
        to answer.  Earlier *groups* are left as the reference gave them, because being asked for
        the next alert after some correct ones is what the profile measures.

        The silence turns are polled exactly as ``_evaluate_turns`` polls them, against the
        annotation's own history and independently of each other.

        ``stop_on_first_failure`` is ignored here.  Stopping would leave the row's other side
        unpolled, and a row that reported one side is not a partial measurement of this mode but a
        different one.
        """
        want_audio = self.use_audio
        predictions: List[dict] = []
        alerts: List[Dict[str, Any]] = []
        media_plan = None
        started = time.time()
        for index, group in enumerate(annotation.alert_groups):
            history: Dict[int, str] = {}
            members: List[dict] = []
            caught: Optional[dict] = None
            for target in group:
                response, diagnostics, request = self._ask(
                    annotation, target.chunk_idx, want_audio, history or None
                )
                media_plan = request.media
                spoke = self.model.dialect.spoke(response)
                prediction = self._prediction(annotation, target, response, spoke)
                if diagnostics:
                    prediction["diagnostics"] = diagnostics
                if self.config.judge and spoke:
                    prediction.update(self.config.judge.judge(annotation, target, response))
                predictions.append(prediction)
                members.append(prediction)
                if spoke:
                    caught = prediction
                    break
                history[target.chunk_idx] = self.model.dialect.as_silence()
            for prediction in members:
                # The alert is the unit, so every second of a caught group passed: a silent second
                # before the one the model spoke on is the grace being used, not a miss.
                prediction["turn_pass"] = caught is not None
            alerts.append({
                "alert_index": index,
                "chunks": [target.chunk_idx for target in group],
                "spoke": caught is not None,
                "first_response_chunk": caught["chunk_idx"] if caught else None,
                # How much of the granted delay was used; 0 on every caught alert of the strict
                # subtask, whose groups are one second wide.
                "response_delay_chunks": (
                    caught["chunk_idx"] - group[0].chunk_idx if caught else None
                ),
            })
        for target in annotation.silence_targets:
            response, diagnostics, request = self._ask(annotation, target.chunk_idx, want_audio)
            media_plan = request.media
            spoke = self.model.dialect.spoke(response)
            prediction = self._prediction(annotation, target, response, spoke)
            prediction["turn_pass"] = not spoke
            if diagnostics:
                prediction["diagnostics"] = diagnostics
            predictions.append(prediction)

        # Polled alerts first and quiet stretches second, but written in stream order: the polls
        # are independent of each other, and a transcript that jumps about is one nobody can read
        # against the video.
        predictions.sort(key=lambda item: item["chunk_idx"])
        judged = [item for item in predictions if item.get("content_ok") is not None]
        content_ok = all(bool(item["content_ok"]) for item in judged) if judged else None
        passed = row_pass(annotation.mode, predictions)
        expected = all(item["spoke"] for item in alerts) and not any(
            item["spoke"] for item in predictions if item.get("alert_index") is None
        )
        if passed != expected:
            # ``row_pass`` re-derives the rule from the turns alone so a saved run can be
            # rescored; if the two ever disagree, one of them is scoring a different benchmark.
            raise AssertionError(
                f"{annotation.sample_id}: row_pass disagrees with the per-alert verdicts"
            )
        record = self._base_record(annotation)
        record.update({
            "pass": passed,
            "content_ok": content_ok,
            "pass_strict": bool(passed and content_ok is not False),
            "alerts": alerts,
            "media": media_plan.as_dict() if media_plan else {},
            "predictions": predictions,
            "latency_sec": round(time.time() - started, 3),
        })
        # Fed the per-alert verdicts, not the per-turn ones, so ``responses_spoken_rate`` counts
        # alerts caught out of alerts and a three-second group is not three chances to score.
        # Added after ``pass`` so a breakdown key cannot overwrite it -- and ``all_responses``
        # is *not* this mode's ``pass``: a row that caught every alert and narrated the gaps
        # between them has failed.
        record.update(response_breakdown(alerts))
        return record

    def _evaluate_running_count(
        self, annotation: Annotation, total_reps: int
    ) -> Dict[str, Any]:
        """Run the row as one serial pass and score the count the model ended up at.

        The only mode whose turns are not independent, and it has to be.  Every other mode polls
        each turn against the annotation's own reference history, which keeps the polls of a row
        from contaminating each other.  Here that history *is* the answer -- the reference says
        "That is 3 times" on the third completion -- so replaying it would hand the model the
        number it is being asked for, and every model would score near 1.0 by reading its own
        prefix.  So the stream is walked once in order and each turn is answered against what
        the model itself said on the turns before it, which is also what a live session would
        have in its context.

        Consequences worth stating, because they are not free:

        * The polls of a row must run in sequence, so ``--workers`` parallelises across rows
          only.  The runner already works that way.
        * An early mistake propagates. That is the measurement: a counter that loses track and
          never recovers has failed the task, and scoring each turn in isolation would hide it.
        * Every chunk after the excluded prefix is polled, including any the row marks as
          neither scored nor excluded.  A skipped chunk would have to be filled from the
          annotation, which is the leak this method exists to avoid.

        The excluded leading chunk is the one exception, and is replayed from the annotation:
        its ground truth is silence in every row, so it leaks nothing, and it is excluded exactly
        because polling it asks a question that is not comparable across repositories.  An
        exclusion anywhere later would leak, so it is refused rather than worked around.

        The content judge is not consulted.  What a reply has to get right here is a number in a
        stated format, which ``read_count`` decides on its own; there is nothing for a judge to
        weigh, and ``content_ok`` reports the format instead.  ``stop_on_first_failure`` is
        likewise ignored: there is no per-turn verdict to stop on, and a truncated pass would
        leave the final count unmeasured.
        """
        want_audio = self.use_audio
        pattern = count_pattern(annotation.reply_pattern)
        scored = {target.chunk_idx: target for target in annotation.targets}
        excluded = {target.chunk_idx for target in annotation.excluded}
        late = sorted(index for index in excluded if index > min(scored))
        if late:
            raise ValueError(
                f"{annotation.sample_id}: chunk {late[0]} is excluded but follows a scored turn; "
                f"in {RUNNING_COUNT} mode its history would have to come from the annotation, "
                "whose reference turns are the count being asked for"
            )

        history: Dict[int, str] = {}
        predictions: List[dict] = []
        media_plan = None
        started = time.time()
        for chunk in annotation.stream:
            if chunk.chunk_idx in excluded:
                continue
            response, diagnostics, request = self._ask(
                annotation, chunk.chunk_idx, want_audio, history
            )
            media_plan = request.media
            # Recorded before anything is read out of it: the next turn is asked against what
            # the model said, verbatim, whether or not a count could be found in it.
            history[chunk.chunk_idx] = response
            spoke = self.model.dialect.spoke(response)
            # A chunk the row scores nothing on still gets a turn, so the pass stays unbroken.
            # ``score_as`` empty is how that reads back: the row stated no expectation here.
            target = scored.get(chunk.chunk_idx) or Target(chunk_idx=chunk.chunk_idx, score_as="")
            prediction = self._prediction(annotation, target, response, spoke)
            if spoke:
                count, format_ok = read_count(response, pattern, self.model.dialect)
                prediction["count"] = count
                prediction["count_format_ok"] = format_ok
            if diagnostics:
                prediction["diagnostics"] = diagnostics
            predictions.append(prediction)

        breakdown = count_breakdown(predictions, total_reps)
        passed = bool(breakdown["count_exact"])
        content_ok = breakdown["count_format_ok"]
        record = self._base_record(annotation)
        record.update({
            "pass": passed,
            # The same field the breakdown carries as ``count_format_ok``, under the name the
            # rest of the harness reports phrasing by.
            "content_ok": content_ok,
            "pass_strict": bool(passed and content_ok is not False),
            "media": media_plan.as_dict() if media_plan else {},
            "predictions": predictions,
            "latency_sec": round(time.time() - started, 3),
        })
        # Added after ``pass`` so that a breakdown key can never overwrite it.
        record.update(breakdown)
        return record
