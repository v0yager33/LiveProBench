#!/usr/bin/env python3
"""LiveProBench evaluation entry point.

    python3 src/run_eval.py --model LiveFlow_omni --suite window_response
    python3 src/run_eval.py --model aura --suite first_event --workers 4

Inputs are named either by suite spec (``profile[:subtask]``, resolved against
the data root) or by explicit ``--input`` paths.  Results land under
``results/<model>/<run>/`` as ``predictions.jsonl`` + ``summary.json`` + ``run.json``, and
because the run name is derived deterministically from what was asked for, re-invoking the
same command with ``--resume`` continues the same run rather than starting a new one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proactive_bench import __version__
from proactive_bench.annotation import SCHEMA
from proactive_bench.conversation import CONTEXT_WINDOW_CHUNKS
from proactive_bench.dataset import write_json
from proactive_bench.extraction import FrameSettings
from proactive_bench.judge import build_judge
from proactive_bench.media import ChunkMedia, default_media_root, ffmpeg_available
from proactive_bench.metrics import format_table, summarize
from proactive_bench.models import add_model_arguments, model_from_args
from proactive_bench.runner import RunConfig, Runner, gather_annotations
from proactive_bench.scoring import EvalConfig, Evaluator
from proactive_bench.suite import (
    COUNT_FRAMES_PER_CHUNK,
    DEFAULT_LAYOUT,
    LAYOUTS,
    InputSpec,
    PROFILES,
    RUNNING_COUNT,
    default_root_for,
    manifest_conflicts,
    resolve_suite,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on LiveProBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--suite", action="append", default=[],
        help="Suite spec 'profile[:subtask]', or 'all'. Repeatable and comma-separated. "
             "Profiles: " + ", ".join(PROFILES) + ".",
    )
    inputs.add_argument("--input", action="append", default=[],
                        help="Explicit JSONL/parquet input. Repeatable.")
    inputs.add_argument("--data-root", default="",
                        help="Root the suite specs are resolved against. Defaults to the root "
                             "the chosen --layout lives under.")
    inputs.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT,
                        help="'delivery' reads the assembled, frozen set; 'generation' "
                             "reads the generator's still-growing output directories.")
    inputs.add_argument("--limit", type=int, default=0,
                        help="Score at most this many rows per input file. 0 means all.")

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    outputs.add_argument("--run-name", default="",
                         help="Run directory name; derived from the inputs when omitted.")
    outputs.add_argument("--output", default="", help="Override the predictions path.")
    outputs.add_argument("--summary-out", default="", help="Override the summary path.")

    run = parser.add_argument_group("run")
    run.add_argument("--workers", type=int, default=4,
                     help="Concurrent rows. Capped by the backend and by the server's "
                          "--max-num-seqs; 3-8 is the tested range for the resident server.")
    run.add_argument("--resume", action="store_true",
                     help="Skip sample ids the predictions file already holds.")
    run.add_argument("--retry-errors", action="store_true",
                     help="With --resume, also retry rows whose only record is a failure.")
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--check", action="store_true",
                     help="Resolve inputs and probe the backend, then exit without scoring.")
    run.add_argument("--context-window", type=int, default=CONTEXT_WINDOW_CHUNKS,
                     help="Seconds of stream one poll may look back over, the way the live "
                          "harnesses these models ship with bound theirs. 0 replays whole "
                          "rows and is only for reproducing a run made before the bound.")

    media = parser.add_argument_group("media")
    media.add_argument("--media-root", default="",
                       help="Extracted frames and audio tracks, as written by "
                            "src/extract_media.py. Defaults to <data-root>/media.")
    media.add_argument("--derive-missing", action="store_true",
                       help="Decode any chunk the media root lacks, instead of stopping. "
                            "Convenient for a few rows; extracting the root first is what "
                            "makes two runs demonstrably see the same pixels.")
    media.add_argument("--frames-per-chunk", type=int, default=1,
                       help="Frames per 1s chunk, i.e. which extracted subtree is read. A "
                            "model whose input geometry fixes this value overrides the flag "
                            "and says so.")
    media.add_argument("--frame-max-side", type=int, default=0,
                       help="Longest side the extracted frames were kept at. 0 means full.")

    judge = parser.add_argument_group("content judge")
    judge.add_argument("--content-judge", action="store_true",
                       help="Also score whether a spoken alert reports the awaited event.")
    judge.add_argument("--judge-provider", default="gemini")
    judge.add_argument("--judge-model", default="")
    judge.add_argument("--judge-slot-root", default="")
    judge.add_argument("--judge-global-slots", type=int, default=0)

    add_model_arguments(parser)
    args = parser.parse_args(argv)
    if not args.suite and not args.input:
        parser.error("give at least one --suite or --input")
    # The layouts live in separate directories, so a --layout delivery that kept the
    # annotation root would resolve to nothing and read as "the set is missing".
    args.data_root = args.data_root or default_root_for(args.layout)
    return args


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "run"


def resolve_inputs(args: argparse.Namespace):
    """Return (specs, missing) for both --suite and --input."""
    specs: List[InputSpec] = []
    missing: List[InputSpec] = []
    if args.suite:
        found, absent = resolve_suite(args.suite, data_root=args.data_root,
                                      layout=args.layout)
        specs.extend(found)
        missing.extend(absent)
    for path in args.input:
        # An explicit file carries its own labels in each row; the spec fields are only a
        # fallback for hand-trimmed debugging files.
        entry = InputSpec(path=path, profile="explicit", subtask="")
        (specs if os.path.isfile(path) else missing).append(entry)
    return specs, missing


def run_name_for(args: argparse.Namespace, model: Any) -> str:
    if args.run_name:
        base = slug(args.run_name)
    elif args.suite:
        base = slug("_".join(",".join(args.suite).split(",")))
    else:
        base = slug(os.path.splitext(os.path.basename(args.input[0]))[0])
    # The audio ablation reads the same rows and writes the same fields, so without a
    # separate directory a --resume would silently graft one run onto the other.  Only the
    # 'off' side is suffixed, which leaves every existing results path untouched -- and only
    # for a model that can actually hear: for a vision-only baseline the flag changes
    # nothing, and a second directory there would present a re-run as an ablation.
    if getattr(args, "audio", "on") == "off" and getattr(model, "has_audio_encoder", False):
        base = f"{base}_no_audio"
    return base


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    for problem in manifest_conflicts(args.data_root):
        # A metric name that has drifted from the delivery manifest would mis-file every
        # rate this run prints, so say so before spending an hour of GPU time on it.
        print(f"[warn] subtask table vs delivery manifest: {problem}", file=sys.stderr)
    specs, missing = resolve_inputs(args)
    for entry in missing:
        # Generation may still be running, so a missing file is reported and skipped rather
        # than aborting a run over everything that does exist.
        print(f"[skip] input not found: {entry.path}", file=sys.stderr)
    if not specs:
        print("no input files exist for the requested suite", file=sys.stderr)
        return 2

    try:
        model = model_from_args(args)
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint reads better than a trace
        print(f"cannot build model {args.model!r}: {exc}", file=sys.stderr)
        return 2
    run_directory = os.path.join(args.results_dir, slug(model.name), run_name_for(args, model))
    if args.audio == "off" and not getattr(model, "has_audio_encoder", False):
        # Silently accepting the flag would leave a run that looks like an ablation but is a
        # duplicate of the ordinary one.
        print(f"note       : {model.name} has no audio encoder; --audio off changes nothing")
    output = args.output or os.path.join(run_directory, "predictions.jsonl")
    summary_path = args.summary_out or os.path.join(run_directory, "summary.json")
    run_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)), "run.json")

    annotations = gather_annotations(specs, limit=args.limit)

    # A model may fix this: ROMA's rope walker allots exactly one temporal grid step per
    # second, and temporal_patch_size 2 makes that exactly two frames.  Overriding the flag
    # rather than erroring keeps `--model roma` correct with no extra ceremony, and the value
    # is written back so run.json records which subtree was actually read.
    required_frames = int(getattr(model, "required_frames_per_chunk", 0) or 0)
    if required_frames and required_frames != args.frames_per_chunk:
        print(
            f"note       : {model.name} requires --frames-per-chunk {required_frames}"
            f" (was {args.frames_per_chunk}); using {required_frames}"
        )
        args.frames_per_chunk = required_frames

    # Said after the override above, so it reports the budget actually served.  Warned rather
    # than refused: the ablation is a legitimate run, and what it must not be is mistaken for
    # the set's own number.  run.json records the budget either way.
    counting = [item for item in annotations if item.mode == RUNNING_COUNT]
    if counting and args.frames_per_chunk < COUNT_FRAMES_PER_CHUNK:
        print(
            f"[warn] {len(counting)} counting row(s) served --frames-per-chunk "
            f"{args.frames_per_chunk}, but their totals were audited at "
            f"{COUNT_FRAMES_PER_CHUNK}: a repetition may not be visible in what is shown, "
            f"and the row still asks for it to be counted",
            file=sys.stderr,
        )

    media = ChunkMedia(
        root=args.media_root or default_media_root(args.data_root),
        settings=FrameSettings(
            frames_per_chunk=args.frames_per_chunk, max_side=args.frame_max_side
        ),
        derive_missing=args.derive_missing,
    )
    print(f"model      : {model.name}")
    print(f"inputs     : {len(specs)} file(s), {len(annotations)} row(s)")
    print(f"media      : {media.root} [{media.settings.tag}]")
    longest = max((len(item.stream) for item in annotations), default=0)
    if args.context_window > 0:
        elided = sum(1 for item in annotations if len(item.stream) > args.context_window)
        print(f"context    : last {args.context_window}s of stream per poll"
              f" ({elided} of {len(annotations)} row(s) longer than that, up to {longest}s)")
    else:
        print(f"context    : unbounded -- whole rows replayed, up to {longest}s")
    print(f"predictions: {output}")

    if args.derive_missing and not ffmpeg_available():
        print("--derive-missing needs ffmpeg on PATH", file=sys.stderr)
        return 2
    # One message up front instead of the same failure repeated once per poll.  Checked over
    # the rows this run will actually score, so a --limit smoke run stats only those -- and
    # ahead of --check, because a check that reports ok while the media root is empty would
    # be answering a different question than the one it is asked.
    if not args.derive_missing:
        gaps = [(item.sample_id, media.missing_chunks(item, want_audio=model.capabilities.audio))
                for item in annotations]
        gaps = [(sample_id, absent) for sample_id, absent in gaps if absent]
        if gaps:
            print(f"{len(gaps)} of {len(annotations)} row(s) have unextracted chunks, e.g. "
                  f"{gaps[0][0]} missing {len(gaps[0][1])} chunk(s) from "
                  f"{gaps[0][1][0]}; extract the media root with\n"
                  f"  python3 src/extract_media.py --suite all --out {media.root}\n"
                  f"or pass --derive-missing to decode as needed", file=sys.stderr)
            model.close()
            return 2

    if args.check:
        probe = getattr(model, "probe", None)
        try:
            if probe is not None:
                print(f"backend    : {probe()}")
        except Exception as exc:  # noqa: BLE001 - an unreachable backend is the answer here
            print(f"backend    : unreachable -- {exc}", file=sys.stderr)
            model.close()
            return 1
        print("check ok")
        model.close()
        return 0

    judge = build_judge(
        args.content_judge,
        provider=args.judge_provider,
        model=args.judge_model,
        slot_root=args.judge_slot_root,
        global_slots=args.judge_global_slots,
    )
    evaluator = Evaluator(
        model,
        EvalConfig(
            media=media,
            judge=judge,
            context_window=args.context_window,
        ),
    )
    runner = Runner(evaluator, RunConfig(
        output=output,
        workers=args.workers,
        limit=args.limit,
        resume=args.resume,
        retry_errors=args.retry_errors,
        progress=not args.no_progress,
    ))

    try:
        records = runner.run(annotations)
    finally:
        model.close()

    summary = summarize(records)
    write_json(summary_path, summary)
    provenance: Dict[str, Any] = {
        "bench_version": __version__,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model.describe(),
        "data_root": args.data_root,
        "layout": args.layout,
        # What was asked for, next to what the model could actually do with it: a row of the
        # audio ablation is only interpretable if both are on record.
        "audio_requested": args.audio,
        "audio_effective": model.capabilities.audio,
        # The silence clause is no longer a run-time choice: it is published in the row's
        # query.text and the set's schema records it.  A run against a v1 set is refused rather
        # than reproduced, so there is nothing here to disambiguate.
        "annotation_schema": SCHEMA,
        # Which interpreter served the run.  A local backend loads its checkpoint from this
        # environment's torch and transformers, so a re-run started from a different one is a
        # different measurement -- and without this field there is no way to find out which.
        "python": sys.executable,
        "inputs": [entry.path for entry in specs],
        "inputs_missing": [entry.path for entry in missing],
        "workers_requested": args.workers,
        "workers_effective": min(args.workers, model.max_concurrency),
        "limit": args.limit,
        "content_judge": judge.describe() if judge else None,
        "media": media.describe(),
        # What each poll could see. A row longer than this is scored on a different prompt than
        # the same row under another bound, so two runs are comparable only at equal values.
        "context_window_chunks": args.context_window,
        "stats": runner.stats.as_dict(),
    }
    write_json(run_path, provenance)

    print()
    print(format_table(summary))
    print()
    print(json.dumps(runner.stats.as_dict(), ensure_ascii=False))
    print(f"summary    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
