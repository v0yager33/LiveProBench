"""Summary aggregation.

The bucket schema is fixed by the handover spec and is deliberately identical for
``overall``, ``by_subtask`` and ``by_profile`` so that a number can be read
the same way wherever it appears.

Two rules the aggregation follows strictly:

* The window-only keys are *absent*, not zero, in a bucket with no window rows.  A zeroed
  ``in_window_rate`` sitting next to a silence subtask's pass rate reads as a real
  measurement, and it is not one.
* Rates are taken over rows that were actually scored.  Rows that failed transport or were
  rejected by the server are counted in ``errors`` and excluded from ``n``, because folding
  them in would quietly report an infrastructure problem as a model failure.

On top of those buckets sit ``by_subset`` and ``headline``, which are what a cross-model table
quotes.  A proactive assistant has to do two opposed things -- speak when the event happens and
stay quiet otherwise -- and either failure alone makes it useless, so a subset pairs the two
sides of its group with a **geometric** mean and the headline averages the subsets.  See
``subsets`` for why the arithmetic mean cannot be used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .annotation import SPEAK
from .suite import (
    RUNNING_COUNT,
    SERIAL,
    SUBSETS,
    SUBTASKS,
    TWO_SIDED_ALERTS,
    WINDOW,
    subset_of,
)

WINDOW_KEYS = (
    "window_rows",
    "verdict_in_window",
    "verdict_early",
    "verdict_missed",
    "in_window_rate",
    "early_rate",
    "missed_rate",
    "mean_response_delay_chunks",
)

# Likewise absent rather than zero outside the repeated-response set.  A row there carries
# three to six reference alerts and answers three questions about them, so one pass rate cannot
# carry it: ``all_responses_rate`` is the headline, ``first_response_rate`` says whether the
# opening alert was caught, and ``two_or_more_responses_rate`` sits between them.  In
# ``two_sided_alerts`` the counted unit is the alert rather than the turn -- a three-second group
# is one alert with three chances, not three alerts -- and ``all_responses_rate`` is no longer
# that mode's ``pass_rate``, which also requires the gaps between the alerts to have stayed quiet.
REPEATED_KEYS = (
    "response_rows",
    "first_response_rate",
    "two_or_more_responses_rate",
    "all_responses_rate",
    "response_turns",
    "responses_spoken",
    "responses_spoken_rate",
)

# Present only for the silence subtasks.  A silence row's target turns are not one continuous
# stretch: the annotation carves out the response window around each reference alert, leaving
# two or more runs of "stay quiet here".  Scoring a whole row all-or-nothing lets one slip in
# the first run mask everything the model did right in the second, so the segment is the unit.
# A ``two_sided_alerts`` row is a silence row too, for the part of it that asks for quiet.
SILENCE_KEYS = (
    "silence_segments",
    "silence_segments_clean",
    "silence_segment_rate",
)

# Present only for the counting subtasks, and absent rather than zero for the same reason as
# the rest: a ``count_exact_rate`` of 0.0 beside a subtask that was never asked to count reads
# as a model that counted wrong.  ``count_exact_rate`` is the headline and equals this mode's
# ``pass_rate``; the others say how a failing row failed -- whether it overshot, undershot, or
# never named a number at all.
COUNT_KEYS = (
    "count_rows",
    "count_exact",
    "count_exact_rate",
    "count_reported_rate",
    "count_over_rate",
    "count_under_rate",
    "mean_abs_count_error",
    "counts_sequential_rate",
)


def _rate(numerator: float, denominator: float) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def _contiguous_runs(predictions: List[dict]) -> List[List[dict]]:
    """Split target turns into maximal runs of consecutive ``chunk_idx``.

    A break in the numbering is not missing data -- it is a stretch the annotation deliberately
    excluded, so the turns either side of it belong to different "keep quiet" obligations.
    """
    runs: List[List[dict]] = []
    ordered = sorted(
        (item for item in predictions if item.get("chunk_idx") is not None),
        key=lambda item: int(item["chunk_idx"]),
    )
    for item in ordered:
        index = int(item["chunk_idx"])
        if runs and index == int(runs[-1][-1]["chunk_idx"]) + 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


@dataclass
class Bucket:
    n: int = 0
    errors: int = 0
    passed: int = 0
    pass_strict: int = 0
    target_turns: int = 0
    target_spoke: int = 0
    content_judged: int = 0
    content_ok: int = 0
    window_rows: int = 0
    in_window: int = 0
    early: int = 0
    missed: int = 0
    delay_sum: int = 0
    delay_n: int = 0
    response_rows: int = 0
    first_response: int = 0
    two_or_more: int = 0
    all_responses: int = 0
    response_turns: int = 0
    responses_spoken: int = 0
    silence_segments: int = 0
    silence_segments_clean: int = 0
    count_rows: int = 0
    count_exact: int = 0
    count_reported: int = 0
    count_over: int = 0
    count_under: int = 0
    count_error_abs_sum: int = 0
    counts_sequential: int = 0

    def add(self, record: dict) -> None:
        if record.get("status") != "ok":
            self.errors += 1
            return
        self.n += 1
        self.passed += int(bool(record.get("pass")))
        self.pass_strict += int(bool(record.get("pass_strict")))

        predictions = record.get("predictions") or []
        self.target_turns += len(predictions)
        self.target_spoke += sum(1 for item in predictions if item.get("spoke"))

        # Judged at row level for the window mode (one alert per row) and at turn level for
        # the serial modes; counting both keeps the ratio a per-decision one.
        verdicts = [
            item.get("content_ok") for item in predictions if item.get("content_ok") is not None
        ]
        if not verdicts and record.get("content_ok") is not None:
            verdicts = [record["content_ok"]]
        self.content_judged += len(verdicts)
        self.content_ok += sum(1 for value in verdicts if value)

        if record.get("verdict"):
            self.window_rows += 1
            verdict = str(record["verdict"])
            if verdict == "in_window":
                self.in_window += 1
            elif verdict == "early":
                self.early += 1
            else:
                self.missed += 1
            delay = record.get("response_delay_chunks")
            if delay is not None:
                self.delay_sum += int(delay)
                self.delay_n += 1

        # Keyed on the mode rather than the subtask name, so a future silence subtask is
        # counted the same way without being named here.  Only the turns that ask for quiet are
        # segmented: every target of a ``serial_target_turns`` row is one, and a two-sided row's
        # alerts are not.  Segmenting those alongside would splice an alert into the quiet stretch
        # beside it and read catching the alert as a broken silence -- failing the model on the
        # silence side for doing exactly what the other side asks.
        if record.get("mode") in (SERIAL, TWO_SIDED_ALERTS):
            quiet = [
                item for item in predictions
                if item.get("expected_action") != SPEAK and item.get("alert_index") is None
            ]
            for segment in _contiguous_runs(quiet):
                self.silence_segments += 1
                self.silence_segments_clean += int(
                    not any(item.get("spoke") for item in segment)
                )

        # Keyed on the breakdown's own field rather than the subtask name, so the numbers
        # follow the mode wherever a future subtask reuses it.
        if record.get("all_responses") is not None:
            self.response_rows += 1
            self.first_response += int(bool(record.get("first_response")))
            self.two_or_more += int(bool(record.get("two_or_more_responses")))
            self.all_responses += int(bool(record.get("all_responses")))
            self.response_turns += int(record.get("response_turns") or 0)
            self.responses_spoken += int(record.get("responses_spoken") or 0)

        # Likewise keyed on the breakdown's own field.  ``count_target`` is the row's ground
        # truth and is written for every counting row, including one that stayed silent all
        # the way through -- which ``final_count`` would not distinguish from a missing key.
        if record.get("count_target") is not None:
            self.count_rows += 1
            self.count_exact += int(bool(record.get("count_exact")))
            if record.get("final_count") is not None:
                self.count_reported += 1
                error = int(record.get("count_error") or 0)
                self.count_error_abs_sum += abs(error)
                if error > 0:
                    self.count_over += 1
                elif error < 0:
                    self.count_under += 1
                # Only among rows that announced something: an empty transcript counts
                # 1..N vacuously, and calling that a clean sequence would flatter silence.
                self.counts_sequential += int(bool(record.get("counts_sequential")))

    def as_dict(self, judge_enabled: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": self.n,
            "errors": self.errors,
            "pass": self.passed,
            "pass_strict": self.pass_strict,
            "pass_rate": _rate(self.passed, self.n),
            "pass_strict_rate": _rate(self.pass_strict, self.n),
            "content_judged": self.content_judged if judge_enabled else None,
            "content_ok": self.content_ok if judge_enabled else None,
            "content_ok_rate": _rate(self.content_ok, self.content_judged) if judge_enabled else None,
            "target_turns": self.target_turns,
            "target_spoke": self.target_spoke,
            "target_spoke_rate": _rate(self.target_spoke, self.target_turns),
        }
        if self.window_rows:
            out.update({
                "window_rows": self.window_rows,
                "verdict_in_window": self.in_window,
                "verdict_early": self.early,
                "verdict_missed": self.missed,
                "in_window_rate": _rate(self.in_window, self.window_rows),
                "early_rate": _rate(self.early, self.window_rows),
                "missed_rate": _rate(self.missed, self.window_rows),
                "mean_response_delay_chunks": _rate(self.delay_sum, self.delay_n),
            })
        if self.response_rows:
            out.update({
                "response_rows": self.response_rows,
                "first_response_rate": _rate(self.first_response, self.response_rows),
                "two_or_more_responses_rate": _rate(self.two_or_more, self.response_rows),
                "all_responses_rate": _rate(self.all_responses, self.response_rows),
                "response_turns": self.response_turns,
                "responses_spoken": self.responses_spoken,
                "responses_spoken_rate": _rate(self.responses_spoken, self.response_turns),
            })
        if self.silence_segments:
            out.update({
                "silence_segments": self.silence_segments,
                "silence_segments_clean": self.silence_segments_clean,
                "silence_segment_rate": _rate(
                    self.silence_segments_clean, self.silence_segments
                ),
            })
        if self.count_rows:
            out.update({
                "count_rows": self.count_rows,
                "count_exact": self.count_exact,
                "count_exact_rate": _rate(self.count_exact, self.count_rows),
                "count_reported_rate": _rate(self.count_reported, self.count_rows),
                "count_over_rate": _rate(self.count_over, self.count_rows),
                "count_under_rate": _rate(self.count_under, self.count_rows),
                # Averaged over the rows that named a number, because a row that named none
                # would need some integer to stand for "no answer" and no integer does.  The
                # unreported rows are visible as the gap in ``count_reported_rate``.
                "mean_abs_count_error": _rate(self.count_error_abs_sum, self.count_reported),
                "counts_sequential_rate": _rate(self.counts_sequential, self.count_reported),
            })
        return out


def _profile_of(subtask: str) -> str:
    item = SUBTASKS.get(subtask)
    return item.profile if item else "unknown"


def _side_of(subtask: str) -> str:
    """Whether a subtask scores speaking up, staying quiet, both, or the two jointly.

    Keyed on the mode so the side follows the scoring rule rather than the subtask's name.
    ``first_response_window`` is ``joint`` because its verdict already fails both ways -- a
    model that never speaks is ``missed`` and one that always speaks is ``early`` -- so it
    needs no silence counterpart to be degeneracy-proof.  ``running_count`` is ``joint`` for
    the same reason and not because it speaks often: silence throughout leaves no count to
    read and fails, and announcing on every second overshoots the total and fails too.

    ``two_sided_alerts`` is ``both``, which is a different thing from ``joint``: a joint mode
    reports one rate that cannot be gamed either way, while a two-sided one reports two rates
    that are then paired -- and the pairing is worth having precisely because both come off the
    same row.
    """
    item = SUBTASKS.get(subtask)
    mode = item.mode if item else ""
    if mode == SERIAL:
        return "silence"
    if mode == TWO_SIDED_ALERTS:
        return "both"
    if mode in (WINDOW, RUNNING_COUNT):
        return "joint"
    return "response"


def _side_rates(subtask: str, bucket: Dict[str, Any]) -> Dict[str, float]:
    """The rate(s) a subtask contributes to its subset, at the finest fixed denominator.

    Deliberately not ``pass_rate`` for the serial modes.  Both of those pool at row level and
    a row is all-or-nothing, which throws away how *much* of the row was right and, worse,
    varies in difficulty with how many turns the row happens to carry.  The per-decision rates
    have denominators fixed by the annotation, so they are comparable across models.

    A mapping rather than the single rate this used to return, because a ``two_sided_alerts``
    subtask supplies both sides out of its one bucket -- alerts caught out of alerts on one side,
    clean quiet stretches out of quiet stretches on the other.  Its ``pass_rate`` is again not
    used, and here for a third reason: it is the conjunction of the two sides, so pairing it with
    itself would report a model's imbalance as if it were its level.
    """
    side = _side_of(subtask)
    if side == "both":
        found = {
            "response": bucket.get("responses_spoken_rate"),
            "silence": bucket.get("silence_segment_rate"),
        }
        return {key: value for key, value in found.items() if value is not None}
    if side == "silence":
        rate = bucket.get("silence_segment_rate")
        return {"silence": rate} if rate is not None else {}
    if side == "joint":
        # Each joint mode has its own denominator and a bucket is per subtask, so exactly one
        # of these is present; the order is not a preference.
        for key in ("in_window_rate", "count_exact_rate"):
            if bucket.get(key) is not None:
                return {"joint": bucket[key]}
        return {}
    rate = bucket.get("responses_spoken_rate")
    if rate is None:
        rate = bucket.get("pass_rate")
    return {"response": rate} if rate is not None else {}


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def subsets(by_subtask: Dict[str, Any]) -> Dict[str, Any]:
    """Pair each subset's speak and silence sides into one score, and average the subsets.

    The pairing is a **geometric** mean, not an arithmetic one, and that is the whole point.
    An arithmetic mean of the two sides is balanced accuracy, under which a degenerate policy
    scores 0.5 rather than 0: a model that never opens its mouth takes 0.0 on the speak side
    and ~1.0 on the silence side and still reads as mid-table.  ``sqrt(speak * silence)``
    collapses when either side collapses, while leaving a model whose two sides are equal with
    exactly the arithmetic mean -- so it penalises imbalance and nothing else.

    Grouped by ``suite.subset_of``, which is the profile for most subtasks and not for
    ``respond_to_anomaly``: it ships under the ``window_response`` profile but scores on its own,
    so that each window subtask weighs the same in the average as ``respond_after_long_silence``
    and ``respond_at_onset`` already do, and so that a ``window_response`` score already published
    keeps meaning the one subtask it was measured on.  The four ``two_sided_alerts`` subtasks are
    grouped apart for the same kind of reason: each pair ships under one profile and its two
    halves ask different questions -- speak on the exact second, or within the seconds granted
    after it -- so one score over a pair would average the grant away.  Each entry carries its
    contributing subtasks for exactly that reason.

    A subset with no silence subtask -- every window one -- scores on its speak side alone and
    is marked ``paired: false``; see ``_side_of`` for why that is not a hole.  A ``two_sided``
    subset is the opposite case: one subtask fills both sides, so it is paired from a single file.
    """
    grouped: Dict[str, Dict[str, List]] = {}
    for name, bucket in by_subtask.items():
        entry = grouped.setdefault(
            subset_of(name), {"response": [], "silence": [], "joint": []}
        )
        for side, rate in _side_rates(name, bucket).items():
            entry[side].append((name, rate))

    out: Dict[str, Any] = {}
    for subset, sides in sorted(grouped.items()):
        speak = _mean([rate for _, rate in sides["response"] + sides["joint"]])
        quiet = _mean([rate for _, rate in sides["silence"]])
        if speak is None and quiet is None:
            continue
        if speak is None or quiet is None:
            score = speak if speak is not None else quiet
            paired = False
        else:
            score = (speak * quiet) ** 0.5
            paired = True
        entry: Dict[str, Any] = {
            "score": score,
            "paired": paired,
            "speak_rate": speak,
            "silence_rate": quiet,
            # The contributing subtasks travel with the score: a subset built from a partial
            # run is a different measurement from the same subset built from a full one, and
            # the number alone cannot be told apart.
            "speak_subtasks": [name for name, _ in sides["response"] + sides["joint"]],
            "silence_subtasks": [name for name, _ in sides["silence"]],
        }
        out[subset] = entry
    return out


def summarize(records: Iterable[dict]) -> Dict[str, Any]:
    """Aggregate prediction records into the overall, subtask, and profile buckets."""
    materialised: List[dict] = list(records)
    # One global decision so a bucket that happens to contain no speak target does not
    # claim the judge was off while its neighbour claims it was on.
    judge_enabled = any(
        record.get("content_ok") is not None
        or any(item.get("content_ok") is not None for item in record.get("predictions") or [])
        for record in materialised
    )

    overall = Bucket()
    by_subtask: Dict[str, Bucket] = {}
    by_profile: Dict[str, Bucket] = {}
    for record in materialised:
        subtask = str(record.get("subtask") or "unknown")
        overall.add(record)
        by_subtask.setdefault(subtask, Bucket()).add(record)
        by_profile.setdefault(_profile_of(subtask), Bucket()).add(record)

    summary: Dict[str, Any] = {"overall": overall.as_dict(judge_enabled)}
    subtasks: Dict[str, Any] = {}
    for name, bucket in sorted(by_subtask.items()):
        entry = bucket.as_dict(judge_enabled)
        item = SUBTASKS.get(name)
        if item:
            # The authoritative metric name travels with the numbers, so a report never has
            # to guess which rate a subtask's headline figure is.
            entry["metric"] = item.metric
            entry["mode"] = item.mode
        subtasks[name] = entry
    summary["by_subtask"] = subtasks
    summary["by_profile"] = {
        name: bucket.as_dict(judge_enabled) for name, bucket in sorted(by_profile.items())
    }
    by_subset = subsets(subtasks)
    summary["by_subset"] = by_subset
    scores = [entry["score"] for entry in by_subset.values() if entry.get("score") is not None]
    summary["headline"] = {
        "avg": _mean(scores),
        "subsets": sorted(by_subset),
        "pairing": "geometric",
    }
    return summary


def headline(summary: Dict[str, Any], subtask: str) -> Optional[float]:
    """The one rate a subtask reports, under its authoritative metric name.

    ``all_responses_rate`` is not checked for here because it is by construction the repeated
    set's ``pass_rate`` -- the mode's pass rule and that breakdown field are the same rule.
    ``count_exact_rate`` is left out for the same reason: the counting mode passes a row iff
    its last count was exact, over the same rows, so the two rates are one number.

    A ``two_sided_alerts`` subtask is the one case where ``pass_rate`` is not the right fallback:
    its metric is the geometric mean of the two sides its bucket measures, while its ``pass_rate``
    is their conjunction taken per row -- a stricter number, under a name that promises this one.
    """
    bucket = (summary.get("by_subtask") or {}).get(subtask)
    if not bucket:
        return None
    item = SUBTASKS.get(subtask)
    if item and item.mode == TWO_SIDED_ALERTS:
        rates = _side_rates(subtask, bucket)
        speak, quiet = rates.get("response"), rates.get("silence")
        if speak is None or quiet is None:
            return None
        return (speak * quiet) ** 0.5
    if "in_window_rate" in bucket:
        return bucket["in_window_rate"]
    return bucket.get("pass_rate")


def _ordered_subsets(by_subset: Dict[str, Any]) -> List[Any]:
    """Subsets in declaration order, which is the order ``src/report.py`` prints them in.

    Alphabetical would put a subset named after a late-added subtask first here and last
    there, and the console block and the cross-model table are read side by side.
    """
    order = {name: index for index, name in enumerate(SUBSETS)}
    return sorted(by_subset.items(),
                  key=lambda item: (order.get(item[0], len(order)), item[0]))


def format_table(summary: Dict[str, Any]) -> str:
    """A fixed-width view of the per-subtask numbers for the console and the run log."""
    rows = (summary.get("by_subtask") or {})
    if not rows:
        return "no scored rows"
    width = max(len(name) for name in rows)
    lines = [
        f"{'subtask'.ljust(width)}  {'n':>5}  {'err':>4}  {'pass_rate':>9}  "
        f"{'strict':>7}  {'spoke':>7}  metric"
    ]
    for name, bucket in rows.items():
        rate = bucket.get("in_window_rate", bucket.get("pass_rate"))
        strict = bucket.get("pass_strict_rate")
        spoke = bucket.get("target_spoke_rate")
        lines.append(
            f"{name.ljust(width)}  {bucket.get('n', 0):>5}  {bucket.get('errors', 0):>4}  "
            f"{'-' if rate is None else format(rate, '.4f'):>9}  "
            f"{'-' if strict is None else format(strict, '.4f'):>7}  "
            f"{'-' if spoke is None else format(spoke, '.4f'):>7}  "
            f"{bucket.get('metric', '')}"
        )
    overall = summary.get("overall") or {}
    lines.append(
        f"{'OVERALL'.ljust(width)}  {overall.get('n', 0):>5}  {overall.get('errors', 0):>4}  "
        f"{'-' if overall.get('pass_rate') is None else format(overall['pass_rate'], '.4f'):>9}"
    )
    # A second block rather than four more columns: these rates exist for one subtask, and
    # widening every row to carry them would print blanks for five of six.
    repeated = {
        name: bucket for name, bucket in rows.items() if bucket.get("response_rows")
    }
    if repeated:
        lines.append("")
        lines.append(
            f"{'repeated alerts'.ljust(width)}  {'rows':>5}  {'first':>7}  {'>=2':>7}  "
            f"{'all':>7}  {'alerts caught':>13}"
        )
        for name, bucket in repeated.items():
            caught = f"{bucket.get('responses_spoken', 0)}/{bucket.get('response_turns', 0)}"
            lines.append(
                f"{name.ljust(width)}  {bucket['response_rows']:>5}  "
                f"{format(bucket.get('first_response_rate') or 0, '.4f'):>7}  "
                f"{format(bucket.get('two_or_more_responses_rate') or 0, '.4f'):>7}  "
                f"{format(bucket.get('all_responses_rate') or 0, '.4f'):>7}  "
                f"{caught:>13}"
            )
    silence = {
        name: bucket for name, bucket in rows.items() if bucket.get("silence_segments")
    }
    if silence:
        lines.append("")
        lines.append(
            f"{'silence'.ljust(width)}  {'segs':>5}  {'clean':>7}  {'seg_rate':>8}  "
            f"{'row_rate':>8}"
        )
        for name, bucket in silence.items():
            lines.append(
                f"{name.ljust(width)}  {bucket['silence_segments']:>5}  "
                f"{bucket.get('silence_segments_clean', 0):>7}  "
                f"{format(bucket.get('silence_segment_rate') or 0, '.4f'):>8}  "
                f"{format(bucket.get('pass_rate') or 0, '.4f'):>8}"
            )
    counting = {
        name: bucket for name, bucket in rows.items() if bucket.get("count_rows")
    }
    if counting:
        lines.append("")
        lines.append(
            f"{'counting'.ljust(width)}  {'rows':>5}  {'exact':>7}  {'over':>7}  "
            f"{'under':>7}  {'silent':>7}  {'|err|':>6}  {'seq':>7}"
        )
        for name, bucket in counting.items():
            # ``silent`` is the complement of ``count_reported_rate``: rows that reached the
            # end of the stream without naming a number at all.  They fail like an overshoot
            # does but for the opposite reason, and the three columns beside it would
            # otherwise leave that gap unaccounted for.
            silent = 1.0 - (bucket.get("count_reported_rate") or 0.0)
            error = bucket.get("mean_abs_count_error")
            lines.append(
                f"{name.ljust(width)}  {bucket['count_rows']:>5}  "
                f"{format(bucket.get('count_exact_rate') or 0, '.4f'):>7}  "
                f"{format(bucket.get('count_over_rate') or 0, '.4f'):>7}  "
                f"{format(bucket.get('count_under_rate') or 0, '.4f'):>7}  "
                f"{format(silent, '.4f'):>7}  "
                f"{'-' if error is None else format(error, '.2f'):>6}  "
                f"{format(bucket.get('counts_sequential_rate') or 0, '.4f'):>7}"
            )
    # The headline block last, because it is the only one whose numbers are not a plain count
    # of rows: each score pairs two sides geometrically, so the sides are printed beside it.
    by_subset = summary.get("by_subset") or {}
    if by_subset:
        lines.append("")
        lines.append(
            f"{'subset'.ljust(width)}  {'speak':>7}  {'silence':>8}  {'score':>7}"
        )
        for name, entry in _ordered_subsets(by_subset):
            # ``-`` rather than 0.0000 for a side that was not run: a subset built from the
            # silence subtask alone scores on silence, and printing its missing speak side as
            # zero would read as a model that never spoke.
            speak = entry.get("speak_rate")
            quiet = entry.get("silence_rate")
            score = entry.get("score")
            lines.append(
                f"{name.ljust(width)}  "
                f"{'-' if speak is None else format(speak, '.4f'):>7}  "
                f"{'-' if quiet is None else format(quiet, '.4f'):>8}  "
                f"{'-' if score is None else format(score, '.4f'):>7}"
            )
        avg = (summary.get("headline") or {}).get("avg")
        lines.append(
            f"{'AVG'.ljust(width)}  {'':>7}  {'':>8}  "
            f"{'-' if avg is None else format(avg, '.4f'):>7}"
        )
    return "\n".join(lines)
