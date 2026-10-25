"""What one annotated second is shown as.

The annotation names a video and which second of it each stream chunk covers; a model is
shown that second as frames plus, when it can hear, an audio track.  Both are files in a
media root that ``src/extract_media.py`` wrote from those timestamps.  Resolution here is a
lookup, not a decode: the run reads files whose checksums are published, so two runs are
demonstrably looking at the same pixels instead of at whatever their own ffmpeg produced.

``derive_missing`` decodes a chunk on demand for convenience, through the same functions the
script uses, so a derived file and a published one are the same bytes.  It is off by default
-- silently spending an hour on ffmpeg is a worse outcome than being told the media root has
not been extracted yet.

A model that declares it can read video is still shown frames.  Frames plus audio is the
benchmark's stream input for every model: handing one of them a 1s mp4 instead would give it
a different decode of the same second, and the resulting numbers would differ by decoder as
much as by model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotation import Annotation, Chunk, Video
from .extraction import (
    DEFAULT_VIDEO_ROOT,
    FROM_CHUNKS,
    ChunkCut,
    FrameSettings,
    MediaError,
    MediaLayout,
    VideoRoots,
    chunk_file,
    extract_audio,
    extract_frames,
    ffmpeg_available,
)

VIDEO = "video"
IMAGE = "image"
AUDIO = "audio"

#: Where an extracted media root sits when the caller names only the annotation root.
MEDIA_DIRNAME = "media"

# Re-exported so callers keep importing media-layer names from the media layer; the one
# definition lives in extraction.py, because two MediaError classes would make an
# ``except MediaError`` in one module blind to a failure raised by the other.
__all__ = [
    "AUDIO", "IMAGE", "VIDEO", "Capabilities", "ChunkMedia", "MEDIA_DIRNAME", "MediaError",
    "MediaPlan", "MediaRef", "default_media_root", "ffmpeg_available",
]


def default_media_root(data_root: str) -> str:
    return os.path.join(data_root, MEDIA_DIRNAME)


@dataclass(frozen=True)
class MediaRef:
    """One concrete media file to hand to a model, in the order it is to be seen."""

    kind: str
    path: str

    def __post_init__(self) -> None:
        if self.kind not in (VIDEO, IMAGE, AUDIO):
            raise ValueError(f"unknown media kind {self.kind!r}")


@dataclass(frozen=True)
class Capabilities:
    """What a model's own input pipeline can ingest."""

    video: bool = False
    image: bool = False
    audio: bool = False

    def accepts(self, kind: str) -> bool:
        return bool(getattr(self, kind))


@dataclass
class MediaPlan:
    """What one row's media actually came to, for the run record.

    The three ways audio can be absent are counted apart, because they mean different
    things when a score is read: the model cannot hear, the run turned audio off, or the
    source video has no audio track at all.
    """

    counts: Dict[str, int] = field(default_factory=dict)
    dropped_audio: int = 0
    missing_audio: int = 0
    derived: int = 0

    def note(self, kind: str, count: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + count

    def as_dict(self) -> Dict[str, int]:
        out = {f"{kind}_parts": count for kind, count in sorted(self.counts.items()) if count}
        if self.dropped_audio:
            out["dropped_audio"] = self.dropped_audio
        if self.missing_audio:
            out["chunks_without_audio_track"] = self.missing_audio
        if self.derived:
            out["derived_on_demand"] = self.derived
        return out


class ChunkMedia:
    """Resolves an annotated second into the files a model is shown.

    One instance serves a whole run: the audio index it consults is loaded once and the
    paths are pure functions of the annotation, so resolution costs a ``stat``.
    """

    def __init__(
        self,
        root: str,
        settings: Optional[FrameSettings] = None,
        video_root: str = DEFAULT_VIDEO_ROOT,
        derive_missing: bool = False,
    ) -> None:
        self.settings = settings or FrameSettings()
        self.layout = MediaLayout(root)
        self.roots = VideoRoots(video_root)
        self.derive_missing = derive_missing
        self._audio_index: Optional[Dict[Tuple[str, str, int], bool]] = None

    @property
    def root(self) -> str:
        return self.layout.root

    # -- paths -----------------------------------------------------------------------

    def frame_paths(self, video: Video, chunk: Chunk) -> List[str]:
        return [
            self.layout.frame_path(
                self.settings, video.dataset, video.video_id, chunk.source_chunk, index
            )
            for index in range(self.settings.frames_per_chunk)
        ]

    def audio_path(self, video: Video, chunk: Chunk) -> str:
        return self.layout.audio_path(video.dataset, video.video_id, chunk.source_chunk)

    # -- the published audio index -------------------------------------------------

    def _audio_tracks(self) -> Dict[Tuple[str, str, int], bool]:
        """Which chunks the extraction found an audio track for.

        Read from the checksum file rather than inferred from a missing ``.wav``: a chunk
        whose source video has no audio stream and a chunk that was never extracted both
        show up as an absent file, and treating the second as the first would quietly feed
        a model silence where it should have heard something.
        """
        if self._audio_index is not None:
            return self._audio_index
        index: Dict[Tuple[str, str, int], bool] = {}
        path = self.layout.checksum_path("audio")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = (
                        str(entry["dataset"]),
                        str(entry["video_id"]),
                        int(entry["source_chunk"]),
                    )
                    index[key] = entry.get("audio") is not None
        self._audio_index = index
        return index

    # -- resolution ------------------------------------------------------------------

    def _derive(self, video: Video, chunk: Chunk, want_audio: bool) -> None:
        """Decode one chunk into the media root, exactly as the extraction script would."""
        chunk_dict = {
            "source_chunk": chunk.source_chunk,
            "source_start": chunk.source_start,
            "source_end": chunk.source_end,
        }
        with chunk_file(chunk_dict, video.dataset, video.video_id, video.relpath, self.roots,
                        FROM_CHUNKS) as path:
            cut = ChunkCut(path=path, seconds=chunk.seconds)
            extract_frames(cut, self.settings, self.frame_paths(video, chunk))
            if want_audio:
                extract_audio(cut, self.audio_path(video, chunk))

    def frames(self, video: Video, chunk: Chunk, plan: Optional[MediaPlan] = None) -> List[str]:
        paths = self.frame_paths(video, chunk)
        if all(os.path.isfile(path) for path in paths):
            return paths
        if not self.derive_missing:
            raise MediaError(
                f"no extracted frames for {video.video_id} chunk {chunk.source_chunk} under "
                f"{self.layout.root} ({self.settings.tag}); run\n"
                f"  python3 src/extract_media.py --suite all --out {self.layout.root}"
                + (f" --frames-per-chunk {self.settings.frames_per_chunk}"
                   if self.settings.frames_per_chunk != 1 else "")
                + "\nor pass --derive-missing to decode chunks as they are needed"
            )
        self._derive(video, chunk, want_audio=True)
        if plan is not None:
            plan.derived += 1
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise MediaError(f"could not derive {os.path.basename(missing[0])} for "
                             f"{video.video_id} chunk {chunk.source_chunk}")
        return paths

    def audio(self, video: Video, chunk: Chunk) -> Optional[str]:
        """This chunk's audio track, or None when the source video has none."""
        path = self.audio_path(video, chunk)
        if os.path.isfile(path):
            return path
        key = (video.dataset, video.video_id, chunk.source_chunk)
        index = self._audio_tracks()
        if key in index:
            # Recorded by the extraction: a False here is a fact about the video, not a gap.
            return path if index[key] else None
        if not self.derive_missing:
            raise MediaError(
                f"no extracted audio for {video.video_id} chunk {chunk.source_chunk} under "
                f"{self.layout.root}; extract the media root with audio, or run with "
                f"--audio off to score this model on frames alone"
            )
        self._derive(video, chunk, want_audio=True)
        return path if os.path.isfile(path) else None

    def refs(
        self,
        video: Video,
        chunk: Chunk,
        caps: Capabilities,
        want_audio: bool,
        plan: MediaPlan,
    ) -> List[MediaRef]:
        """The files for one chunk turn, frames first and the track last."""
        refs = [MediaRef(IMAGE, path) for path in self.frames(video, chunk, plan)]
        plan.note(IMAGE, len(refs))
        if not (want_audio and caps.audio):
            if want_audio:
                # Asked for, but this model has no audio encoder to put it through.
                plan.dropped_audio += 1
            return refs
        track = self.audio(video, chunk)
        if track is None:
            plan.missing_audio += 1
            return refs
        refs.append(MediaRef(AUDIO, track))
        plan.note(AUDIO)
        return refs

    # -- preflight -------------------------------------------------------------------

    def missing_chunks(self, annotation: Annotation, want_audio: bool = True) -> List[int]:
        """Which of a row's chunks have no extracted media yet.

        Used before a run starts, so that an unextracted media root is one message up front
        rather than the same failure repeated once per poll.
        """
        absent: List[int] = []
        tracks = self._audio_tracks()
        for chunk in annotation.stream:
            paths = self.frame_paths(annotation.video, chunk)
            if not all(os.path.isfile(path) for path in paths):
                absent.append(chunk.source_chunk)
                continue
            if not want_audio:
                continue
            key = (annotation.video.dataset, annotation.video.video_id, chunk.source_chunk)
            if key not in tracks and not os.path.isfile(self.audio_path(annotation.video, chunk)):
                absent.append(chunk.source_chunk)
        return absent

    def describe(self) -> Dict[str, object]:
        return {
            "root": self.layout.root,
            "frames": self.settings.tag,
            "frames_per_chunk": self.settings.frames_per_chunk,
            "derive_missing": self.derive_missing,
        }
