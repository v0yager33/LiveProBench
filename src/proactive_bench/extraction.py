"""Deterministic derivation of frames and audio from the benchmark's videos.

An annotation row carries no pixels and no samples: it names a source video and, per stream
chunk, the second of that video the chunk shows.  Everything a model is shown is *derived*
from those timestamps, which is what makes the set publishable -- but it also means that two
people running the benchmark only see the same images if they derive them the same way.  This
module is that one way, and :mod:`extract_media` is its command line.

The derivation is two steps, and the intermediate step matters:

1. one second of source video becomes a 1s ``chunk_NNNN.mp4``, with the command the dataset's
   own chunker used (:func:`chunk_command`);
2. that chunk file becomes frames and an audio track (:func:`extract_frames`,
   :func:`extract_audio`).

Step 1 looks redundant when the chunk tree is already on disk -- and in ``chunks`` mode it is
skipped -- but it is what makes the set reproducible from the original corpus alone: on one
ffmpeg build the command rebuilds each chunk file *byte for byte* (verified with
``extract_media.py --compare``), so a third party who only has ActivityNetQA can regenerate
the exact intermediate and therefore the exact frames.

It also cannot be replaced by seeking the source directly at ``source_start + offset``.  A
re-encoded cut keeps the frame that covers its seek instant, so the chunk file's own timeline
sits up to one frame ahead of the source's; sampling both at their 0.5s mark then picks
neighbouring frames, which on fast motion is a visibly different image (measured on
ActivityNetQA: SSIM ~0.84 in the worst sampled second, against ~0.98 for the same frame
through one extra re-encode).  ``--compare`` reports that number rather than leaving it as an
assumption, and direct seeking is deliberately not one of the extraction modes.

``-bitexact`` is passed when extracting so that no encoder version string is muxed into a
``.wav`` header, which would otherwise change every checksum on an ffmpeg upgrade while the
samples stayed the same.  It is deliberately *not* passed when rebuilding a chunk, because
there the goal is to match the existing tree, which was written without it.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

# Each cut is one second, so ffmpeg's default thread count buys nothing and just
# oversubscribes the box when many extractions run in parallel.
FFMPEG_THREADS = os.environ.get("PROACTIVE_BENCH_FFMPEG_THREADS", "2")

CHUNK_SECONDS = 1.0
FRAME_QUALITY = 2
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

# How the dataset's own chunker encoded a second of video.  Changing any of these values
# changes the pixels every model is scored on, so they are constants, not options.
CHUNK_VIDEO_CODEC = "libx264"
CHUNK_VIDEO_PRESET = "ultrafast"
CHUNK_VIDEO_CRF = "23"
CHUNK_AUDIO_CODEC = "aac"

FROM_CHUNKS = "chunks"
FROM_REBUILD = "rebuild"
SOURCE_MODES = (FROM_CHUNKS, FROM_REBUILD)

DEFAULT_VIDEO_ROOT = os.environ.get(
    "PROACTIVE_BENCH_VIDEO_ROOT",
    "/chatgpt_nas/dukaixuan.dkx/LiveChatOmni/data/datasets/LiveChatOmni",
)

REPRODUCIBILITY_NOTE = (
    "media is derived in two steps -- one second of source video becomes a 1s chunk mp4 with "
    "the dataset's own encoder settings, and that chunk becomes frames and audio -- so a "
    "third party holding only the original videos can regenerate the intermediate byte for "
    "byte on the same ffmpeg build and therefore the same frames; 'extracted_from' says "
    "whether this tree read the shipped chunk files or rebuilt them, and both produce "
    "identical output, while seeking the source video directly does not and is not offered"
)


class MediaError(RuntimeError):
    """A chunk's media cannot be produced in the form the model needs."""


@dataclass(frozen=True)
class FrameSettings:
    """How many frames a chunk becomes, and how large they are kept.

    ``frames_per_chunk`` is a model-visible quantity -- ROMA's rope walker allots exactly two
    frames per second -- so a media root holds one subtree per setting rather than one
    subtree that silently means different things to different models.
    """

    frames_per_chunk: int = 1
    max_side: int = 0

    def __post_init__(self) -> None:
        if self.frames_per_chunk < 1:
            raise ValueError("frames_per_chunk must be at least 1")
        if self.max_side < 0:
            raise ValueError("max_side cannot be negative")

    @property
    def tag(self) -> str:
        """Directory name for this setting: every value that changes a pixel is in it."""
        side = f"s{self.max_side}" if self.max_side > 0 else "full"
        return f"n{self.frames_per_chunk}_{side}"

    def offsets(self) -> List[float]:
        """Sampling offsets inside a one-second chunk.

        Midpoints of equal slices, never slice boundaries: the first frame of a cut second is
        frequently a decode artefact of the cut itself.
        """
        count = self.frames_per_chunk
        return [(index + 0.5) / count for index in range(count)]

    def as_dict(self) -> Dict[str, object]:
        return {
            "tag": self.tag,
            "frames_per_chunk": self.frames_per_chunk,
            "max_side": self.max_side,
            "jpeg_quality": FRAME_QUALITY,
            "offsets_seconds": [round(value, 6) for value in self.offsets()],
        }


@dataclass(frozen=True)
class VideoRoots:
    """The two video trees an annotation's ``video.relpath`` is resolved against."""

    root: str = DEFAULT_VIDEO_ROOT

    def source_video(self, relpath: str) -> str:
        return os.path.join(self.root, "videos", relpath)

    def chunk_video(self, dataset: str, video_id: str, source_chunk: int) -> str:
        return os.path.join(
            self.root, "video_chunks", dataset, video_id, f"chunk_{int(source_chunk):04d}.mp4"
        )


@dataclass(frozen=True)
class ChunkCut:
    """One second of media, as a file plus an offset inside it.

    ``start`` is zero for a chunk file and non-zero only for the direct-source cut that
    ``--compare`` measures; keeping the offset explicit is what lets the extraction commands
    be written once for both.
    """

    path: str
    start: float = 0.0
    seconds: float = CHUNK_SECONDS

    def frame_timestamp(self, offset: float) -> float:
        return self.start + offset * self.seconds


def chunk_seconds_of(chunk: dict) -> float:
    """The span one annotated chunk covers, checked against the layout's assumption."""
    seconds = float(chunk["source_end"]) - float(chunk["source_start"])
    if abs(seconds - CHUNK_SECONDS) > 1e-6:
        raise MediaError(
            f"chunk {chunk.get('source_chunk')} spans {seconds}s; the extraction layout "
            f"assumes {CHUNK_SECONDS}s chunks"
        )
    return seconds


def chunk_command(source: str, start: float, seconds: float) -> List[str]:
    """The dataset chunker's own command for one second, minus the output path.

    Reproduced verbatim on purpose.  Adding ``-bitexact``, or pinning ``-threads``, would
    produce a defensible file that no longer matches the chunk tree already shipped, and the
    whole point of this command is that it does.
    """
    return [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", source, "-t", f"{seconds:.3f}",
        "-c:v", CHUNK_VIDEO_CODEC, "-preset", CHUNK_VIDEO_PRESET, "-crf", CHUNK_VIDEO_CRF,
        "-c:a", CHUNK_AUDIO_CODEC, "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CHANNELS),
    ]


def chunk_settings() -> Dict[str, object]:
    return {
        "seconds": CHUNK_SECONDS,
        "video_codec": CHUNK_VIDEO_CODEC,
        "preset": CHUNK_VIDEO_PRESET,
        "crf": CHUNK_VIDEO_CRF,
        "audio_codec": CHUNK_AUDIO_CODEC,
        "audio_rate": AUDIO_RATE,
        "audio_channels": AUDIO_CHANNELS,
    }


@contextlib.contextmanager
def chunk_file(
    chunk: dict,
    dataset: str,
    video_id: str,
    relpath: str,
    roots: VideoRoots,
    mode: str = FROM_CHUNKS,
    workdir: str = "",
    keep: bool = False,
) -> Iterator[str]:
    """Yield the 1s chunk file for one annotated second.

    In ``chunks`` mode this is the shipped file and nothing is written.  In ``rebuild`` mode it
    is regenerated from the source video into ``workdir``; unless ``keep`` is set the rebuilt
    file is deleted on the way out, because keeping every intermediate costs about a gigabyte
    per thousand chunks and the frames are the artefact anyone actually needs.
    """
    if mode not in SOURCE_MODES:
        raise ValueError(f"unknown source mode {mode!r}")
    source_chunk = int(chunk["source_chunk"])
    if mode == FROM_CHUNKS:
        path = roots.chunk_video(dataset, video_id, source_chunk)
        if not os.path.isfile(path):
            raise MediaError(
                f"chunk file missing: {path}; run with --from rebuild to regenerate it from "
                f"{roots.source_video(relpath)}"
            )
        yield path
        return

    if not workdir:
        raise ValueError("rebuild mode needs a workdir to write the rebuilt chunk into")
    seconds = chunk_seconds_of(chunk)
    source = roots.source_video(relpath)
    if not os.path.isfile(source):
        raise MediaError(f"source video missing: {source}")
    command = chunk_command(source, float(chunk["source_start"]), seconds)
    if keep:
        target = os.path.join(workdir, dataset, video_id, f"chunk_{source_chunk:04d}.mp4")
        produced = produce(target, command)
        if not produced:
            raise MediaError(f"could not rebuild chunk {source_chunk} of {video_id}")
        yield produced
        return
    # A private name per worker: the file is transient, so two workers rebuilding the same
    # second must not be able to read each other's half-written mp4.
    os.makedirs(workdir, exist_ok=True)
    target = os.path.join(
        workdir,
        f"{video_id}.{source_chunk:04d}.{os.getpid()}.{threading.get_ident()}.mp4",
    )
    try:
        run_ffmpeg([*command, target])
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            raise MediaError(f"could not rebuild chunk {source_chunk} of {video_id}")
        yield target
    finally:
        if os.path.exists(target):
            os.remove(target)


def direct_source_cut(chunk: dict, relpath: str, roots: VideoRoots) -> ChunkCut:
    """The cut a naive implementation would make: seek the source and sample there.

    Only ``--compare`` uses this, to measure how far it lands from the chunk-file path.  It is
    not an extraction mode precisely because that distance is not zero.
    """
    return ChunkCut(
        path=roots.source_video(relpath),
        start=float(chunk["source_start"]),
        seconds=chunk_seconds_of(chunk),
    )


class MediaLayout:
    """Paths inside a media root.

    Frames live under their setting's tag so several frame rates can coexist; audio does not
    depend on the frame setting and is stored once.
    """

    def __init__(self, root: str) -> None:
        self.root = root

    def frames_dir(self, settings: FrameSettings, dataset: str, video_id: str,
                   source_chunk: int) -> str:
        return os.path.join(
            self.root, "frames", settings.tag, dataset, video_id,
            f"chunk_{int(source_chunk):04d}",
        )

    def frame_path(self, settings: FrameSettings, dataset: str, video_id: str,
                   source_chunk: int, index: int) -> str:
        return os.path.join(
            self.frames_dir(settings, dataset, video_id, source_chunk), f"frame_{index:02d}.jpg"
        )

    def audio_path(self, dataset: str, video_id: str, source_chunk: int) -> str:
        return os.path.join(
            self.root, "audio", dataset, video_id, f"chunk_{int(source_chunk):04d}.wav"
        )

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def checksum_path(self, name: str) -> str:
        return os.path.join(self.root, "checksums", f"{name}.jsonl")

    def relative(self, path: str) -> str:
        return os.path.relpath(path, self.root)


_LOCKS: Dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _LOCK_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = _LOCKS[path] = threading.Lock()
        return lock


def run_ffmpeg(command: Sequence[str]) -> None:
    result = subprocess.run(
        list(command), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"ffmpeg failed: {detail[-1] if detail else 'no stderr'}")


def produce(target: str, command: Sequence[str]) -> Optional[str]:
    """Run ``command`` into ``target`` unless it already exists, atomically.

    Written to a staging name and renamed, under a per-target lock: many workers can be
    asked for the same file at once, and without this one of them reads a half-written JPEG.
    """
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    with _lock_for(target):
        if os.path.exists(target) and os.path.getsize(target) > 0:
            return target
        os.makedirs(os.path.dirname(target), exist_ok=True)
        root, extension = os.path.splitext(target)
        # The staging name keeps the target's extension: ffmpeg picks its output muxer from
        # it, and a bare ".tmp.<pid>.<tid>" tail makes it refuse the write.
        staging = f"{root}.tmp.{os.getpid()}.{threading.get_ident()}{extension}"
        try:
            run_ffmpeg([*command, staging])
            if not os.path.exists(staging) or os.path.getsize(staging) == 0:
                return None
            os.replace(staging, target)
        finally:
            if os.path.exists(staging):
                os.remove(staging)
    return target


def _scale_filter(max_side: int) -> List[str]:
    if max_side <= 0:
        return []
    return [
        "-vf",
        f"scale='min({max_side},iw)':'min({max_side},ih)'"
        ":force_original_aspect_ratio=decrease",
    ]


def frame_command(cut: ChunkCut, offset: float, settings: FrameSettings) -> List[str]:
    """The exact ffmpeg invocation for one frame, minus the output path."""
    return [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-threads", FFMPEG_THREADS,
        "-fflags", "+bitexact",
        "-ss", f"{cut.frame_timestamp(offset):.6f}", "-i", cut.path,
        *_scale_filter(settings.max_side),
        "-map", "0:v:0", "-frames:v", "1", "-q:v", str(FRAME_QUALITY),
        "-bitexact", "-f", "image2",
    ]


def audio_command(cut: ChunkCut) -> List[str]:
    """The exact ffmpeg invocation for one chunk's audio, minus the output path."""
    return [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-threads", FFMPEG_THREADS,
        "-fflags", "+bitexact",
        "-ss", f"{cut.start:.6f}", "-i", cut.path, "-t", f"{cut.seconds:.6f}",
        "-vn", "-map", "0:a:0", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE),
        "-c:a", AUDIO_CODEC, "-bitexact", "-f", "wav",
    ]


def extract_frames(cut: ChunkCut, settings: FrameSettings, targets: Sequence[str]) -> List[str]:
    """Produce this chunk's frames. Raises when not one frame could be decoded."""
    if len(targets) != settings.frames_per_chunk:
        raise ValueError("one target path per frame is required")
    produced: List[str] = []
    for offset, target in zip(settings.offsets(), targets):
        path = produce(target, frame_command(cut, offset, settings))
        if path:
            produced.append(path)
    if not produced:
        raise MediaError(f"could not extract any frame from {cut.path} at {cut.start}s")
    return produced


def extract_audio(cut: ChunkCut, target: str) -> Optional[str]:
    """Produce this chunk's audio track, or None when the video has no audio stream.

    A trackless video is normal in a vision corpus and must not fail a row; the caller
    records the absence instead.
    """
    try:
        return produce(target, audio_command(cut))
    except MediaError:
        return None


def sha256_file(path: str, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(block), b""):
            digest.update(piece)
    return digest.hexdigest()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_build() -> str:
    """The ffmpeg banner, recorded so a checksum mismatch can be attributed to a build."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "unavailable"
    line = result.stdout.decode("utf-8", "replace").splitlines()
    return line[0].strip() if line else "unknown"
