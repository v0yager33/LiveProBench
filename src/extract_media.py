#!/usr/bin/env python3
"""Turn the annotation's timestamps into the frames and audio a model is shown.

    python3 src/extract_media.py --suite all
    python3 src/extract_media.py --suite all --from rebuild
    python3 src/extract_media.py --verify
    python3 src/extract_media.py --suite all --compare 32

The published set is annotation only: it names a video and, per stream chunk, the second of
that video the chunk shows.  This script is the other half of that contract -- the one
documented way to turn those seconds into pixels and samples, so that a run somewhere else is
looking at the same images and hearing the same audio as a run here rather than at whatever
its own decoder happened to produce.

What it writes, under ``--out``::

    frames/n1_full/<dataset>/<video_id>/chunk_0009/frame_00.jpg
    audio/<dataset>/<video_id>/chunk_0009.wav
    checksums/frames_n1_full.jsonl
    checksums/audio.jsonl
    manifest.json

Frames are keyed by their setting (``n1_full`` is one full-size frame per chunk) because
frames-per-chunk is model-visible -- ROMA is fed two frames per second -- so several settings
have to be able to coexist without one quietly standing in for another.  Audio does not
depend on that setting and is stored once.

``--from chunks`` (the default) reads the 1s chunk files shipped with the corpus.  ``--from
rebuild`` regenerates each of those files from the original video first, for anyone who has
ActivityNetQA but not our chunk tree; the two produce identical frames because the rebuild
command reproduces the chunk file itself.  ``--compare`` is what establishes that: it rebuilds
a sample and diffs the mp4s byte for byte, and -- because the tempting shortcut is to skip the
intermediate entirely -- also reports how far a direct seek into the source lands from the
chunk file, which is up to a full frame and visibly so.  ``--verify`` re-hashes an existing
tree against its own checksums, which is how a third party establishes that they are looking
at our images and not merely at images.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proactive_bench import annotation as ann
from proactive_bench.dataset import write_json
from proactive_bench.extraction import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_RATE,
    DEFAULT_VIDEO_ROOT,
    FROM_CHUNKS,
    FROM_REBUILD,
    REPRODUCIBILITY_NOTE,
    SOURCE_MODES,
    ChunkCut,
    FrameSettings,
    MediaError,
    MediaLayout,
    VideoRoots,
    chunk_file,
    chunk_settings,
    direct_source_cut,
    extract_audio,
    extract_frames,
    ffmpeg_available,
    ffmpeg_build,
    sha256_file,
)
from proactive_bench.suite import (
    ANNOTATION,
    DEFAULT_ANNOTATION_ROOT,
    LAYOUTS,
    PROFILES,
    resolve_suite,
)

MEDIA_SCHEMA = "proactive_bench_media_v1"
SSIM_RE = re.compile(r"All:([0-9.]+)")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive LiveProBench frames and audio from annotated timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--suite", action="append", default=[],
                        help="Suite spec 'profile[:subtask]', or 'all'. Profiles: "
                             + ", ".join(PROFILES))
    inputs.add_argument("--input", action="append", default=[],
                        help="Explicit annotation JSONL. Repeatable.")
    inputs.add_argument("--data-root", default=DEFAULT_ANNOTATION_ROOT)
    inputs.add_argument("--layout", choices=LAYOUTS, default=ANNOTATION,
                        help="Only the annotation layout carries the timestamps this needs.")
    inputs.add_argument("--limit", type=int, default=0, help="Rows per file. 0 means all.")
    inputs.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT,
                        help="Holds videos/<relpath> and video_chunks/<dataset>/<video_id>/.")

    output = parser.add_argument_group("output")
    output.add_argument("--out", default="",
                        help="Media root. Defaults to <data-root>/media.")
    output.add_argument("--source", "--from", dest="source", choices=SOURCE_MODES,
                        default=FROM_CHUNKS,
                        help="'chunks' reads the shipped 1s chunk files; 'rebuild' "
                             "regenerates each of them from the original video first.")
    output.add_argument("--keep-rebuilt", default="", metavar="DIR",
                        help="With --from rebuild, keep the regenerated chunk mp4s here "
                             "instead of discarding them (about 1 GB per 1000 chunks).")
    output.add_argument("--frames-per-chunk", type=int, default=1)
    output.add_argument("--frame-max-side", type=int, default=0,
                        help="Downscale frames to this longest side. 0 keeps full size.")
    output.add_argument("--no-audio", action="store_true", help="Frames only.")
    output.add_argument("--workers", type=int, default=12, help="Parallel ffmpeg calls.")

    checks = parser.add_argument_group("checks")
    checks.add_argument("--verify", action="store_true",
                        help="Re-hash the existing tree against its checksums and exit.")
    checks.add_argument("--compare", type=int, default=0, metavar="N",
                        help="Extract N chunks both ways and report how far they diverge.")
    args = parser.parse_args(argv)
    if not args.verify and not args.suite and not args.input:
        parser.error("give at least one --suite or --input")
    return args


# -- planning -------------------------------------------------------------------------


class Job:
    """One chunk of one video to derive media for."""

    __slots__ = ("dataset", "video_id", "relpath", "chunk")

    def __init__(self, dataset: str, video_id: str, relpath: str, chunk: ann.Chunk) -> None:
        self.dataset = dataset
        self.video_id = video_id
        self.relpath = relpath
        self.chunk = chunk

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.dataset, self.video_id, self.chunk.source_chunk)

    def as_chunk_dict(self) -> Dict[str, object]:
        return {
            "source_chunk": self.chunk.source_chunk,
            "source_start": self.chunk.source_start,
            "source_end": self.chunk.source_end,
        }


def collect_files(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.suite:
        present, missing = resolve_suite(args.suite, data_root=args.data_root,
                                         layout=args.layout)
        for entry in missing:
            print(f"[skip] input not found: {entry.path}", file=sys.stderr)
        paths.extend(entry.path for entry in present)
    for path in args.input:
        if os.path.isfile(path):
            paths.append(path)
        else:
            print(f"[skip] input not found: {path}", file=sys.stderr)
    return paths


def plan(paths: Sequence[str], limit: int = 0) -> Tuple[List[Job], int]:
    """Every distinct chunk the given annotation files reference, and the row count.

    De-duplicated on ``(dataset, video_id, source_chunk)``: rows may share a source video,
    and extracting the same second once per row would multiply the work by the overlap.
    """
    jobs: Dict[Tuple[str, str, int], Job] = {}
    rows = 0
    for path in paths:
        for row in ann.load(path, limit=limit):
            rows += 1
            for chunk in row.stream:
                job = Job(row.video.dataset, row.video.video_id, row.video.relpath, chunk)
                jobs.setdefault(job.key, job)
    return [jobs[key] for key in sorted(jobs)], rows


# -- extraction -----------------------------------------------------------------------


def derive(job: Job, roots: VideoRoots, layout: MediaLayout, settings: FrameSettings,
           mode: str, want_audio: bool, workdir: str, keep: bool) -> Dict[str, object]:
    """Produce one chunk's media and return its checksum records."""
    record: Dict[str, object] = {
        "dataset": job.dataset,
        "video_id": job.video_id,
        "source_chunk": job.chunk.source_chunk,
        "source_start": round(job.chunk.source_start, 6),
    }
    targets = [
        layout.frame_path(settings, job.dataset, job.video_id, job.chunk.source_chunk, index)
        for index in range(settings.frames_per_chunk)
    ]
    audio_target = layout.audio_path(job.dataset, job.video_id, job.chunk.source_chunk)
    with chunk_file(job.as_chunk_dict(), job.dataset, job.video_id, job.relpath, roots,
                    mode, workdir=workdir, keep=keep) as path:
        cut = ChunkCut(path=path, start=0.0, seconds=job.chunk.seconds)
        frames = extract_frames(cut, settings, targets)
        track = extract_audio(cut, audio_target) if want_audio else None
    record["frames"] = [
        {
            "path": layout.relative(item),
            "sha256": sha256_file(item),
            "bytes": os.path.getsize(item),
        }
        for item in frames
    ]
    if want_audio:
        record["audio"] = (
            {
                "path": layout.relative(track),
                "sha256": sha256_file(track),
                "bytes": os.path.getsize(track),
            }
            if track
            else None
        )
    return record


def read_checksums(path: str) -> Dict[Tuple[str, str, int], dict]:
    known: Dict[Tuple[str, str, int], dict] = {}
    if not os.path.isfile(path):
        return known
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            known[(entry["dataset"], entry["video_id"], int(entry["source_chunk"]))] = entry
    return known


def write_checksums(path: str, entries: Dict[Tuple[str, str, int], dict]) -> None:
    """Rewrite a checksum file, sorted, so its own digest is a property of the content.

    Merged with what was already there rather than replaced: a ``--limit`` pass must not
    silently drop the chunks a previous full pass recorded.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for key in sorted(entries):
            handle.write(json.dumps(entries[key], ensure_ascii=False) + "\n")


def extract(args: argparse.Namespace, paths: List[str]) -> int:
    settings = FrameSettings(args.frames_per_chunk, args.frame_max_side)
    roots = VideoRoots(args.video_root)
    layout = MediaLayout(args.out)
    want_audio = not args.no_audio
    keep = bool(args.keep_rebuilt)

    jobs, rows = plan(paths, args.limit)
    print(f"annotations: {len(paths)} file(s), {rows} row(s)")
    print(f"chunks     : {len(jobs)} distinct second(s) of video")
    print(f"from       : {args.source}"
          + (f", kept in {args.keep_rebuilt}" if keep else ""))
    print(f"frames     : {settings.tag} at offsets "
          f"{[round(value, 3) for value in settings.offsets()]}")
    print(f"audio      : {'off' if not want_audio else f'{AUDIO_RATE} Hz mono {AUDIO_CODEC}'}")
    print(f"out        : {args.out}")

    frame_entries = read_checksums(layout.checksum_path(f"frames_{settings.tag}"))
    audio_entries = read_checksums(layout.checksum_path("audio")) if want_audio else {}
    failures: List[str] = []
    silent = 0
    with tempfile.TemporaryDirectory(prefix=".rebuild_", dir=_scratch_parent(args)) as scratch:
        workdir = args.keep_rebuilt or scratch
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(derive, job, roots, layout, settings, args.source, want_audio,
                            workdir, keep): job
                for job in jobs
            }
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                job = futures[future]
                try:
                    record = future.result()
                except MediaError as exc:
                    failures.append(f"{job.video_id} chunk {job.chunk.source_chunk}: {exc}")
                    continue
                key = job.key
                frame_entries[key] = {
                    "dataset": record["dataset"],
                    "video_id": record["video_id"],
                    "source_chunk": record["source_chunk"],
                    "source_start": record["source_start"],
                    "frames": record["frames"],
                }
                if want_audio:
                    audio = record.get("audio")
                    if audio is None:
                        silent += 1
                    audio_entries[key] = {
                        "dataset": record["dataset"],
                        "video_id": record["video_id"],
                        "source_chunk": record["source_chunk"],
                        "audio": audio,
                    }
                if done % 2000 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)} chunk(s) derived", flush=True)

    for problem in failures[:20]:
        print(f"  [fail] {problem}", file=sys.stderr)
    if len(failures) > 20:
        print(f"  ... {len(failures) - 20} more failure(s)", file=sys.stderr)

    frames_path = layout.checksum_path(f"frames_{settings.tag}")
    write_checksums(frames_path, frame_entries)
    if want_audio:
        write_checksums(layout.checksum_path("audio"), audio_entries)

    write_manifest(args, layout, settings, paths, rows, frame_entries, audio_entries,
                   want_audio, silent, len(failures))
    print(f"manifest   : {layout.manifest_path}")
    if failures:
        print(f"{len(failures)} chunk(s) could not be derived", file=sys.stderr)
        return 1
    return 0


def _scratch_parent(args: argparse.Namespace) -> str:
    """Where an extraction pass puts its transient rebuilt chunks.

    Inside the media root rather than the system temp dir: the rebuild writes about a megabyte
    per chunk and a full pass would fill a small ``/tmp`` long before it finished.  Only the
    extraction pass needs this; ``--compare`` rebuilds a handful and stays in the system temp
    dir, so asking for a comparison does not leave a media root behind.
    """
    os.makedirs(args.out, exist_ok=True)
    return args.out


def _totals(entries: Dict[Tuple[str, str, int], dict], key: str) -> Tuple[int, int]:
    files = 0
    total_bytes = 0
    for entry in entries.values():
        items = entry.get(key)
        if items is None:
            continue
        for item in items if isinstance(items, list) else [items]:
            files += 1
            total_bytes += int(item.get("bytes") or 0)
    return files, total_bytes


def write_manifest(args, layout: MediaLayout, settings: FrameSettings, paths: List[str],
                   rows: int, frame_entries: dict, audio_entries: dict, want_audio: bool,
                   silent: int, failures: int) -> None:
    """Merge this pass into the media root's manifest.

    Merged, not overwritten: a root legitimately holds several frame settings, and the second
    pass must not erase the first one's record of itself.
    """
    manifest = {}
    if os.path.isfile(layout.manifest_path):
        with open(layout.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    frames_path = layout.checksum_path(f"frames_{settings.tag}")
    files, size = _totals(frame_entries, "frames")
    variants = dict(manifest.get("frames") or {})
    variants[settings.tag] = {
        **settings.as_dict(),
        "chunks": len(frame_entries),
        "files": files,
        "bytes": size,
        "checksums": layout.relative(frames_path),
        "checksums_sha256": sha256_file(frames_path),
        "failed_chunks": failures,
    }
    manifest.update({
        "schema": MEDIA_SCHEMA,
        "extracted_from": args.source,
        "video_root": args.video_root,
        "annotation_root": args.data_root,
        "annotation_files": sorted(os.path.basename(path) for path in paths),
        "annotation_rows": rows,
        "ffmpeg": ffmpeg_build(),
        "chunk": chunk_settings(),
        "frames": variants,
        "reproducibility_note": REPRODUCIBILITY_NOTE,
    })
    if want_audio:
        audio_path = layout.checksum_path("audio")
        audio_files, audio_bytes = _totals(audio_entries, "audio")
        manifest["audio"] = {
            "sample_rate": AUDIO_RATE,
            "channels": AUDIO_CHANNELS,
            "codec": AUDIO_CODEC,
            "chunks": len(audio_entries),
            "files": audio_files,
            "bytes": audio_bytes,
            "chunks_without_audio_stream": silent,
            "checksums": layout.relative(audio_path),
            "checksums_sha256": sha256_file(audio_path),
        }
    write_json(layout.manifest_path, manifest)


# -- verification ---------------------------------------------------------------------


def verify(args: argparse.Namespace) -> int:
    """Re-hash every file a checksum file names, and report what has drifted."""
    layout = MediaLayout(args.out)
    if not os.path.isfile(layout.manifest_path):
        print(f"no media manifest at {layout.manifest_path}", file=sys.stderr)
        return 2
    with open(layout.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    print(f"root       : {args.out}")
    print(f"built from : {manifest.get('extracted_from')} with {manifest.get('ffmpeg')}")

    problems = 0
    checked = 0
    for name in list((manifest.get("frames") or {}).keys()):
        entry = manifest["frames"][name]
        got, bad = _verify_group(layout, entry.get("checksums"), "frames")
        checked += got
        problems += bad
        print(f"frames {name}: {got} file(s) checked, {bad} mismatched or missing")
    audio = manifest.get("audio")
    if audio:
        got, bad = _verify_group(layout, audio.get("checksums"), "audio")
        checked += got
        problems += bad
        print(f"audio      : {got} file(s) checked, {bad} mismatched or missing")

    print(f"total      : {checked} file(s), {problems} problem(s)")
    if problems:
        print(
            "a mismatch with a different ffmpeg build than the manifest records is expected "
            "and means re-derived, not wrong; a mismatch on the same build means the tree "
            "has been edited",
            file=sys.stderr,
        )
        return 1
    return 0


def _verify_group(layout: MediaLayout, checksums: Optional[str], key: str) -> Tuple[int, int]:
    if not checksums:
        return 0, 0
    path = os.path.join(layout.root, checksums)
    checked = 0
    problems = 0
    reported = 0
    for entry in read_checksums(path).values():
        items = entry.get(key)
        if items is None:
            continue
        for item in items if isinstance(items, list) else [items]:
            checked += 1
            target = os.path.join(layout.root, item["path"])
            if not os.path.isfile(target):
                problems += 1
                if reported < 10:
                    reported += 1
                    print(f"  [missing] {item['path']}", file=sys.stderr)
                continue
            if sha256_file(target) != item["sha256"]:
                problems += 1
                if reported < 10:
                    reported += 1
                    print(f"  [changed] {item['path']}", file=sys.stderr)
    return checked, problems


# -- cross-source comparison -----------------------------------------------------------


def frame_ssim(left: str, right: str) -> Optional[float]:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", left, "-i", right, "-lavfi", "ssim", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    match = SSIM_RE.search(result.stderr.decode("utf-8", "replace"))
    return float(match.group(1)) if match else None


def audio_correlation(left: str, right: str) -> Optional[float]:
    """Pearson correlation of two mono 16-bit tracks, over their common length."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        samples = []
        for path in (left, right):
            with wave.open(path, "rb") as handle:
                raw = handle.readframes(handle.getnframes())
            samples.append(np.frombuffer(raw, dtype="<i2").astype("float64"))
    except (OSError, wave.Error, EOFError):
        return None
    length = min(len(samples[0]), len(samples[1]))
    if length < 2:
        return None
    left_signal, right_signal = samples[0][:length], samples[1][:length]
    if left_signal.std() == 0 or right_signal.std() == 0:
        return None
    value = float(np.corrcoef(left_signal, right_signal)[0, 1])
    return None if math.isnan(value) else value


def compare(args: argparse.Namespace, paths: List[str]) -> int:
    """Establish that the media is reproducible, and price the shortcut that isn't.

    Two questions, per sampled chunk:

    * does rebuilding the 1s chunk from the original video reproduce the shipped file?  If it
      does, everything downstream of it is reproducible too, and that is the whole claim the
      published set makes about its media.
    * how far off is a direct seek into the source at the same instant?  This is the obvious
      shortcut, so its cost is measured rather than left for someone to rediscover.
    """
    settings = FrameSettings(args.frames_per_chunk, args.frame_max_side)
    roots = VideoRoots(args.video_root)
    jobs, _ = plan(paths, args.limit)
    if not jobs:
        print("nothing to compare", file=sys.stderr)
        return 2
    wanted = max(1, min(int(args.compare), len(jobs)))
    # Evenly spaced rather than the first N: the first N are all the same few videos, and a
    # seek bug that only shows up late in a long file would go unseen.
    step = len(jobs) / wanted
    sample = [jobs[min(len(jobs) - 1, int(index * step))] for index in range(wanted)]
    print(f"comparing  : {len(sample)} chunk(s) with {ffmpeg_build()}")

    identical = 0
    differing: List[str] = []
    ssims: List[float] = []
    correlations: List[float] = []
    with tempfile.TemporaryDirectory(prefix="pb_compare_") as work:
        for index, job in enumerate(sample):
            chunk = job.as_chunk_dict()
            shipped = roots.chunk_video(job.dataset, job.video_id, job.chunk.source_chunk)
            layout = MediaLayout(os.path.join(work, "media"))
            targets = [
                layout.frame_path(settings, job.dataset, job.video_id,
                                  job.chunk.source_chunk, position)
                for position in range(settings.frames_per_chunk)
            ]
            audio_target = layout.audio_path(job.dataset, job.video_id, job.chunk.source_chunk)
            try:
                with chunk_file(chunk, job.dataset, job.video_id, job.relpath, roots,
                                FROM_REBUILD, workdir=os.path.join(work, "rebuild")) as rebuilt:
                    same = os.path.isfile(shipped) and (
                        sha256_file(rebuilt) == sha256_file(shipped)
                    )
                    cut = ChunkCut(path=rebuilt, seconds=job.chunk.seconds)
                    chunk_frame = extract_frames(cut, settings, targets)[0]
                    chunk_audio = (
                        None if args.no_audio else extract_audio(cut, audio_target)
                    )
                # The naive alternative: same instant, straight out of the source video.
                direct = direct_source_cut(chunk, job.relpath, roots)
                direct_layout = MediaLayout(os.path.join(work, "direct"))
                direct_frame = extract_frames(
                    direct, settings,
                    [
                        direct_layout.frame_path(settings, job.dataset, job.video_id,
                                                 job.chunk.source_chunk, position)
                        for position in range(settings.frames_per_chunk)
                    ],
                )[0]
                direct_audio = (
                    None
                    if args.no_audio
                    else extract_audio(
                        direct,
                        direct_layout.audio_path(job.dataset, job.video_id,
                                                 job.chunk.source_chunk),
                    )
                )
            except MediaError as exc:
                print(f"  [fail] {job.video_id} chunk {job.chunk.source_chunk}: {exc}",
                      file=sys.stderr)
                continue

            if same:
                identical += 1
            else:
                differing.append(f"{job.video_id} chunk {job.chunk.source_chunk}")
            ssim = frame_ssim(chunk_frame, direct_frame)
            correlation = (
                audio_correlation(chunk_audio, direct_audio)
                if chunk_audio and direct_audio
                else None
            )
            if ssim is not None:
                ssims.append(ssim)
            if correlation is not None:
                correlations.append(correlation)
            if index < 10 or not same:
                print(f"  {job.video_id} chunk {job.chunk.source_chunk:>4}: "
                      f"rebuild={'identical' if same else 'DIFFERS'} "
                      f"direct_seek_ssim={'n/a' if ssim is None else f'{ssim:.4f}'} "
                      f"direct_seek_audio_corr="
                      f"{'n/a' if correlation is None else f'{correlation:.4f}'}")

    def report(name: str, values: List[float]) -> None:
        if not values:
            print(f"{name}: no measurement")
            return
        print(f"{name}: n={len(values)} min={min(values):.4f} "
              f"mean={sum(values) / len(values):.4f} max={max(values):.4f}")

    print(f"rebuild    : {identical}/{len(sample)} chunk mp4(s) reproduced byte for byte")
    report("direct seek frame ssim", ssims)
    report("direct seek audio corr", correlations)
    if differing:
        for name in differing[:10]:
            print(f"  [differs] {name}", file=sys.stderr)
        print(
            "a chunk that does not reproduce means this ffmpeg/x264 build encodes differently "
            "from the one that built the shipped tree; extract with --from chunks to stay on "
            "the shipped files, and record this build in the run",
            file=sys.stderr,
        )
    print(REPRODUCIBILITY_NOTE)
    return 1 if differing else 0


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    args.out = args.out or os.path.join(args.data_root, "media")
    if not ffmpeg_available():
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2
    if args.verify:
        return verify(args)
    paths = collect_files(args)
    if not paths:
        print("no annotation files exist", file=sys.stderr)
        return 2
    if args.compare:
        return compare(args, paths)
    return extract(args, paths)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
