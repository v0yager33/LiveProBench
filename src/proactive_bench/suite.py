"""Subtask registry and dataset-path resolution.

``data_generation/build_eval_proactive.py``'s ``TASK_META`` owns the authoritative
metric names; the table below mirrors it and the two must stay in sync.  Every caller
resolves a ``--suite`` spec through here so no script hardcodes a dataset path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ``scoring.mode`` values.  Dispatch on these, never on a long chain of subtask names.
LAST_TURN = "last_turn"
SERIAL = "serial_target_turns"
WINDOW = "first_response_window"
# The one mode whose polls are not independent: the row is run as a single serial pass and
# each turn is answered against the model's *own* earlier replies rather than the
# annotation's.  It has to be, because the annotation's reference turns are the answer --
# replaying "That is 3 times" into the history would hand a counting model its own count.
RUNNING_COUNT = "running_count"
# Both sides of one row.  Every other mode asks a row a single question -- speak here, or stay
# quiet there -- so the paired score in ``metrics.subsets`` has to draw its two rates from two
# different files, which means two different videos: the geometric mean then pairs a model's
# speaking on one recording against its silence on another, and a row's own behaviour is never
# scored both ways.  A ``two_sided_alerts`` row carries both kinds of target, so both rates come
# off the same stream and the pairing is about one model on one recording.  Targets that want
# speech are grouped by ``alert_index`` and a group counts as caught if the model opens its mouth
# on any second in it; how wide that group is belongs to the row, not to the scorer -- one second
# for the strict subtask, three for the grace one -- so the two subtasks share this mode and
# differ only in their data, the same way ``respond_at_onset``'s grant lives in the generator.
TWO_SIDED_ALERTS = "two_sided_alerts"
#: Frames per 1s chunk the counting rows' totals were audited at.  Every repetition in that
#: set was confirmed distinguishable at two frames a second, and the rows whose period that
#: budget could not resolve were dropped rather than kept.  Serving fewer frames than this can
#: hide a completion the row still asks to be counted, which arrives as an undercount that
#: looks like a model failure.
COUNT_FRAMES_PER_CHUNK = 2

# Four on-disk layouts exist and they are not interchangeable.
#
# ``annotation`` ``<data_root>/annotations/<subtask>.jsonl`` -- the published set: pure
#                annotation, no model's messages and no media, so every model assembles its
#                own prompt from it.  This is what the benchmark is, hence the default.
# ``test``       ``<data_root>/test/<publication_name>.jsonl`` -- publication-ready test set
#                with cleaned schema (no internal fields, simplified IDs). File names match
#                paper terminology rather than internal subtask names.
# ``delivery``   ``<data_root>/delivery/<subtask>.jsonl`` -- the generator's assembled rows,
#                still carrying LiveChatOmni's own message list.  Kept resolvable because
#                the annotation set is built from it and the two must stay comparable.
# ``generation`` the generator's own output directories. Background generation is still
#                running against these, so row counts move under your feet; useful for
#                looking at the newest rows, wrong for a number anyone will quote.
ANNOTATION = "annotation"
TEST = "test"
DELIVERY = "delivery"
GENERATION = "generation"
LAYOUTS: Tuple[str, ...] = (ANNOTATION, TEST, DELIVERY, GENERATION)
DEFAULT_LAYOUT = ANNOTATION

# Mapping from internal subtask names to publication-ready file names (for TEST layout)
# Matches paper terminology: ES, OR, WR, AA, SS, DC
SUBTASK_TO_PUBLICATION_NAME = {
    "respond_only_at_first_evidence": "event_sensitivity",      # ES
    "respond_at_onset": "onset_response",                      # OR
    "respond_within_window": "window_response",                # WR
    "respond_to_anomaly": "anomaly_alert",                     # AA
    "respond_after_long_silence": "sustained_silence",         # SS
    "count_every_occurrence": "duplicate_counting",            # DC
}

DEFAULT_ANNOTATION_ROOT = os.environ.get(
    "PROACTIVE_BENCH_DATA_ROOT",
    "/chatgpt_nas/dukaixuan.dkx/LiveChatOmni/data/ProactiveBench",
)
DEFAULT_DELIVERY_ROOT = os.environ.get(
    "PROACTIVE_BENCH_DELIVERY_ROOT",
    "/chatgpt_nas/dukaixuan.dkx/LiveChatOmni/data/eval_proactive_clear_targets_v2",
)
DEFAULT_DATA_ROOT = DEFAULT_ANNOTATION_ROOT


def default_root_for(layout: str = DEFAULT_LAYOUT) -> str:
    """The root a layout is normally found under.

    The layouts live in separate directories, so a ``--layout delivery`` that kept the
    annotation root would resolve to nothing and read as "the set is missing".
    """
    if layout in (ANNOTATION, TEST):
        return DEFAULT_ANNOTATION_ROOT
    return DEFAULT_DELIVERY_ROOT


@dataclass(frozen=True)
class Subtask:
    """One subtask's identity, and the two groupings it belongs to.

    ``profile`` is the on-disk grouping and is not ours to choose: it is recorded in every
    row, in the delivery and annotation manifests, and in the builders, so changing one
    would mean rebuilding the set.  ``subset`` is the scoring grouping -- which pairing a
    subtask's rate is averaged into for the headline -- and defaults to the profile.  The two
    are separate because a subtask can be delivered alongside another and still be its own
    measurement: see ``metrics.subsets``.
    """

    name: str
    profile: str
    mode: str
    metric: str
    pass_rule: str
    subset: str = ""
    #: The sentence appended to every row's query so the model is told the silence rule it is
    #: scored against. Empty would mean the subtask asks for no silence, which none do.  Baked
    #: into the published ``query.text``; kept here as the authority a rebuild copies from.
    hold_clause: str = ""


# Every query asks to be told when something happens and none of them asked for quiet before it
# did, while the scorer fails any speech on a turn whose reference is silence.  Rows were
# therefore being lost to a rule the model was never given: narrating the wait, or promising to
# report later, on turns that only silence could pass.  These clauses state that rule in the
# user's own voice.  They are benchmark content and identical for every model, so they belong in
# this table beside ``pass_rule`` rather than in any ``Dialect`` -- a per-model wording would
# make the runs incomparable.  None of them says when the event is due, so nothing of the answer
# leaks; what they add is the requirement, not the timing.
HOLD_UNTIL_EVENT = "Until then, stay silent -- no running commentary, no progress updates."
HOLD_BETWEEN_ALERTS = (
    "Between those alerts, stay silent -- no running commentary, no progress updates."
)
HOLD_UNLESS_WRONG = (
    "As long as nothing goes wrong, stay silent -- no running commentary, no progress updates."
)
# The counting subtasks ask to be told at *every* completion rather than once, so neither
# "until then" nor "tell me once" fits: what they need held is the gap between completions.
# "At every other moment" said that, but also reads as every second completion, which is a
# different counting rule from the one being scored -- so the gap is named directly instead.
HOLD_BETWEEN_COUNTS = (
    "The rest of the time, stay silent -- no running commentary, no progress updates."
)


_SUBTASKS: Tuple[Subtask, ...] = (
    Subtask(
        "respond_only_at_first_evidence", "first_event", TWO_SIDED_ALERTS,
        "first_evidence_caught_gmean",
        "One row carries both sides of the sharpest timing question in the suite. The alert group "
        "is the single second the evidence first appears, so the model has to speak on that exact "
        "second and no grace is granted; the seconds in front of it are the same row's silence "
        "targets, held to two independent judgements -- the plan's own account of why nothing is "
        "reportable there, and a sweep of every one of those seconds against the query, which "
        "drops the row if any of them already answers it. The one second between the two sides is "
        "streamed and never polled -- the plan declined to rule on that boundary, so neither side "
        "may claim it. The score reported is the geometric mean of the alert-caught rate and the "
        "silence-segment rate, both measured on this one file.",
        subset="first_evidence",
        hold_clause=HOLD_UNTIL_EVENT,
    ),
    # ``recover_after_missed_evidence`` stood here and is retired from the scored table.  It was the
    # same footage as the subtask above, cut two seconds further and asked the opposite way round: the
    # reference is silent on the second the evidence appears, so the alert group is the two seconds
    # after a miss that already happened.  Every other subtask in the suite asks for nothing before
    # the response point and the awaited thing plainly there at it, and this one asks for the reverse
    # -- its judged seconds continue an action that began before them, by construction, on every row.
    # That cost it a private exemption in each gate that reads the rows, and the exemptions do not
    # compose: a reader told not to mind that something started earlier is the same reader that has to
    # catch an alert fired before anything started.  Its build profile is untouched, being the one
    # ``respond_only_at_first_evidence`` is also cut from, and its rows and audits are kept under
    # ``data/LiveProBench/retired/``.
    # ``response_at_first_evidence`` and ``no_early_response`` stood here and are retired from the
    # scored table, for the same reason as the prior-responses pair below: one scored a single
    # spoken second and the other the silence in front of one, over two disjoint halves of the
    # same video pool, so pairing them set a model's speaking on one set of recordings against its
    # silence on another.  The two subtasks above ask both questions of one row, and the half that
    # used to be scored for silence alone is what ``recover_after_missed_evidence`` is cut from.
    # Their build profile is untouched -- it produced the plans both are re-cut from, and is how
    # the pool is topped up.  ``last_turn`` and ``serial_target_turns`` are left with no live
    # subtask by this; both modes stay in ``scoring`` regardless, because the runs already
    # published under them have to remain re-scorable.
    # ``response_after_prior_responses`` and ``no_response_after_prior_responses`` stood here and
    # are retired from the scored table, the way ``no_post_response`` was: they asked this
    # profile's two questions of two different halves of the video pool, so pairing them scored a
    # model's speaking on one set of recordings against its silence on another.  The two subtasks
    # below ask both of one row.  Their build profile is untouched -- it produced the plans these
    # are re-cut from, and is how the pool is topped up.
    Subtask(
        "respond_at_every_alert", "repeated_response", TWO_SIDED_ALERTS,
        "alerts_caught_at_onset_gmean",
        "One row carries both sides. Score every alert group and every retained silent stretch "
        "of the same stream, each with the dataset's own history. An alert group is a single "
        "second here, so the model has to speak on the exact second the reference alerts on. "
        "The row passes iff every alert is caught and every silent stretch stays quiet, and the "
        "score reported for it is the geometric mean of the alert-caught rate and the "
        "silence-segment rate -- both measured on this one file.",
        subset="repeated_strict",
        hold_clause=HOLD_BETWEEN_ALERTS,
    ),
    # ``respond_within_grace_every_alert`` stood here and is retired from the scored table.  It was
    # ``respond_at_every_alert`` with each alert group three seconds wide instead of one, the extra two
    # granted as recognition delay against a pinned alert point.  The distinction stopped being one:
    # both gates that clear these rows read only the first second of a group and treat the rest as
    # latency rather than evidence, so the two files put the same question to a model at two widths and
    # the wider one is the easier half of a measurement already reported by the strict one.  Its rows
    # and audits are kept under ``data/LiveProBench/retired/``.
    Subtask(
        "respond_within_window", "window_response", WINDOW,
        "first_response_in_window_rate",
        "Poll every chunk turn after the first with an all-silent history. Pass iff the first "
        "non-silent turn falls inside the response window.",
        hold_clause=HOLD_UNTIL_EVENT,
    ),
    Subtask(
        "respond_after_long_silence", "long_report", WINDOW,
        "first_response_in_window_rate_after_long_silence",
        "Poll every chunk turn after the first of a long recording with an all-silent history. "
        "The window opens only after a verified silent prefix of over a minute. Pass iff the "
        "first non-silent turn falls inside the response window.",
        hold_clause=HOLD_UNTIL_EVENT,
    ),
    Subtask(
        "respond_at_onset", "immediate_response", WINDOW,
        "first_response_at_onset_rate",
        "Poll every chunk turn after the first and require the first non-silent one to fall "
        "within the three seconds that open at the annotated onset: a second earlier is early, "
        "a second past the third is missed. Three gates pin that onset to the exact second the "
        "request becomes unmistakable, so the width is recognition delay granted against a "
        "boundary that is known, not a vaguer event given a longer run.",
        hold_clause=HOLD_UNTIL_EVENT,
    ),
    Subtask(
        "respond_to_anomaly", "window_response", WINDOW,
        "first_response_on_anomaly_rate",
        "Poll every chunk turn after the first under one standing request shared by every row "
        "of this subtask. Pass iff the first non-silent turn falls inside the window that opens "
        "when the mishap becomes visible and closes when a first report would no longer be "
        "timely.",
        # Delivered under the ``window_response`` profile but scored as its own subset. It asks a
        # different question from ``respond_within_window`` -- one standing request against a
        # seconds-long mishap, rather than a per-video request against a sustained activity --
        # so folding the two into one score would average that difference away and, worse,
        # silently redefine a ``window_response`` figure that has already been published.
        subset="anomaly_alert",
        hold_clause=HOLD_UNLESS_WRONG,
    ),
    Subtask(
        "count_every_occurrence", "dedup_counting", RUNNING_COUNT,
        "total_count_exact_rate",
        "Run the row's chunk turns as one serial pass, each answered against the model's own "
        "earlier replies, and read the counts it announced. Pass iff the last count announced "
        "equals the row's total_reps; a row that never announces one fails. Whether every "
        "announcement used the requested wording is reported beside it as content_ok, and "
        "whether the counts ran 1..N without a repeat is reported as a diagnostic. What is "
        "counted is whatever the row's own query names -- a repetition of an exercise, or an "
        "object-handling event in a scene that also holds distractor objects handled in ways "
        "that do not count -- which changes the corpus a row came from, not the question asked "
        "of the model or the number it is scored on.",
        hold_clause=HOLD_BETWEEN_COUNTS,
    ),
)

SUBTASKS: Dict[str, Subtask] = {item.name: item for item in _SUBTASKS}
PROFILES: Tuple[str, ...] = ("first_event", "repeated_response", "window_response", "long_report",
                             "immediate_response", "dedup_counting")


def subset_of(subtask: str) -> str:
    """Which subset a subtask's rate is averaged into for the headline.

    The profile unless the subtask names its own, so the default stays "one subset per
    profile" and an override has to be written down deliberately.
    """
    item = SUBTASKS.get(subtask)
    if not item:
        return "unknown"
    return item.subset or item.profile


# Declaration order, so a report that walks this prints the subsets in the order the subtasks
# were introduced rather than alphabetically.
SUBSETS: Tuple[str, ...] = tuple(
    dict.fromkeys(item.subset or item.profile for item in _SUBTASKS)
)


def profile_subtasks(profile: str) -> List[Subtask]:
    return [item for item in _SUBTASKS if item.profile == profile]


def subset_subtasks(subset: str) -> List[Subtask]:
    return [item for item in _SUBTASKS if subset_of(item.name) == subset]


def profile_dir(profile: str, data_root: str = DEFAULT_DATA_ROOT) -> str:
    """Directory holding one profile's generated subtask files."""
    if profile == "first_event":
        return data_root
    return os.path.join(data_root, profile)


def input_path(
    profile: str,
    subtask: str,
    data_root: str = DEFAULT_DATA_ROOT,
    layout: str = DEFAULT_LAYOUT,
) -> str:
    """Where one subtask's rows live under the requested layout."""
    if layout == ANNOTATION:
        return os.path.join(data_root, "annotations", f"{subtask}.jsonl")
    if layout == TEST:
        # Map internal subtask name to publication file name
        pub_name = SUBTASK_TO_PUBLICATION_NAME.get(subtask, subtask)
        return os.path.join(data_root, "test", f"{pub_name}.jsonl")
    if layout == DELIVERY:
        return os.path.join(data_root, DELIVERY, f"{subtask}.jsonl")
    if layout == GENERATION:
        return os.path.join(profile_dir(profile, data_root), f"{subtask}.jsonl")
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def manifest_path(data_root: str = DEFAULT_DATA_ROOT, layout: str = DEFAULT_LAYOUT) -> str:
    if layout == ANNOTATION:
        return os.path.join(data_root, "manifest.json")
    return os.path.join(data_root, DELIVERY, "manifest.json")


def load_manifest(data_root: str = DEFAULT_DATA_ROOT, layout: str = DEFAULT_LAYOUT) -> Dict:
    """The set's index, or ``{}`` when it has not been built."""
    path = manifest_path(data_root, layout)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest_conflicts(data_root: str = DEFAULT_DATA_ROOT,
                       layout: str = DEFAULT_LAYOUT) -> List[str]:
    """Disagreements between the table above and the set's own manifest.

    The manifest carries each file's ``metric`` and ``mode`` as the builder recorded
    them.  If our table has drifted from it, every rate this harness prints is filed under
    the wrong name -- worth catching loudly rather than discovering in a paper table.
    """
    manifest = load_manifest(data_root, layout)
    problems: List[str] = []
    for name, entries in (manifest.get("subtasks") or {}).items():
        item = SUBTASKS.get(name)
        if not item:
            problems.append(f"{name}: in the manifest but not in the subtask table")
            continue
        # The delivery manifest lists one entry per source file; the annotation manifest
        # carries a single object per subtask.  Both are read here so the check survives
        # whichever set a run is pointed at.
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            for key, ours in (("metric", item.metric), ("mode", item.mode)):
                theirs = entry.get(key)
                if theirs and theirs != ours:
                    problems.append(
                        f"{name}: manifest {key}={theirs!r} but table says {ours!r}"
                    )
    return sorted(set(problems))



@dataclass(frozen=True)
class InputSpec:
    """One JSONL file to evaluate, plus the labels it is expected to carry."""

    path: str
    profile: str
    subtask: str

    @property
    def slug(self) -> str:
        return f"{self.profile}.{self.subtask}"


def _parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) > 2:
        raise ValueError(f"suite spec {spec!r} must be profile[:subtask]")
    parts += [""] * (2 - len(parts))
    profile, subtask = parts
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r} in suite spec {spec!r}; expected one of {PROFILES}"
        )
    if subtask and subtask not in SUBTASKS:
        raise ValueError(f"unknown subtask {subtask!r} in suite spec {spec!r}")
    if subtask and SUBTASKS[subtask].profile != profile:
        raise ValueError(f"subtask {subtask!r} does not belong to profile {profile!r}")
    return profile, subtask or None


def resolve_suite(
    specs: Sequence[str],
    data_root: str = DEFAULT_DATA_ROOT,
    layout: str = DEFAULT_LAYOUT,
) -> Tuple[List[InputSpec], List[InputSpec]]:
    """Expand ``--suite`` specs into concrete files.

    Returns ``(present, missing)``. Missing files are returned rather than raised because
    generation may still be running: a run over what exists today must not be blocked by a
    subtask file that has not been produced yet.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    wanted: List[str] = []
    for spec in specs:
        for piece in str(spec).split(","):
            piece = piece.strip()
            if piece:
                wanted.append(piece)
    if not wanted:
        raise ValueError("empty suite spec")
    if any(piece == "all" for piece in wanted):
        wanted = [profile for profile in PROFILES]

    present: List[InputSpec] = []
    missing: List[InputSpec] = []
    seen: set[str] = set()
    for piece in wanted:
        profile, subtask = _parse_spec(piece)
        subtasks = [SUBTASKS[subtask]] if subtask else profile_subtasks(profile)
        for item in subtasks:
            path = input_path(profile, item.name, data_root, layout)
            if path in seen:
                continue
            seen.add(path)
            entry = InputSpec(path, profile, item.name)
            (present if os.path.isfile(path) else missing).append(entry)
    return present, missing


def describe_subtask(name: str) -> Dict[str, str]:
    item = SUBTASKS.get(name)
    if not item:
        return {"subtask": name}
    return {
        "subtask": item.name,
        "profile": item.profile,
        # Both groupings, because they are not always the same one and a reader of the output
        # cannot tell which score a subtask landed in from the profile alone.
        "subset": subset_of(item.name),
        "mode": item.mode,
        "metric": item.metric,
        "pass_rule": item.pass_rule,
        "hold_clause": item.hold_clause,
    }


def metric_name(subtask: str) -> str:
    item = SUBTASKS.get(subtask)
    return item.metric if item else ""


def ask_with_hold(text: str, subtask: str) -> str:
    """The query as a row publishes it: the asked text, then the subtask's silence clause.

    The one place this join is written.  Two upstream steps produce the set -- one converts the
    delivery rows, one transforms what is already published -- and a second copy of the rule
    would let the two disagree about a row's prompt, which is the prompt the run is scored
    against.  The harness itself no longer joins anything: it reads ``query.text`` as published.

    A period is supplied when the query does not end in one, so the clause reads as its own
    sentence rather than running on.  Idempotent, so a set can be passed through twice without
    growing a second clause.
    """
    item = SUBTASKS.get(subtask)
    clause = item.hold_clause if item else ""
    text = text.rstrip()
    if not clause or text.endswith(clause):
        return text
    if text and text[-1] not in ".!?":
        text = f"{text}."
    return f"{text} {clause}"
