"""ROMA baseline -- the one model whose silence decision is not text.

Every other baseline answers the polling question in words: it either emits an alert or it
emits a silence token.  ROMA does not.  Its checkpoint carries a binary "speak now?" gate
(``thinker.gate_head*`` together with ``thinker.gate_mixer``) that is read off the hidden
states at the assistant prompt position, and the language head only runs once that gate has
fired.  Asking ROMA for text at every poll would measure the wrong thing: its silent turns
are masked out of the LM loss during training, so the model was never taught to *say*
``<|silence|>``.  This adapter reproduces the model's own decision procedure instead --
read the gate, compare against a threshold, decode text only when it fires -- and reports
the raw probability through :meth:`generate_with_diagnostics` so a finished run can be
rescored at another threshold without touching the GPU again.  ROMA's own proactive demo
stops at the probability and never decodes at all (``gradio/proactive_gradio.py``); text is
decoded here because the suite scores what was said, and because the LM head *was* trained
on the turns that speak -- on targets that are the literal word ``alert`` in ROMA's
proactive data, which is why its content is not comparable to a model that answers in
sentences.  See ``docs/baselines.md``.

``<|silence|>`` is not a borrowed marker on the *input* side: ROMA's own preprocessing
writes that exact literal into every silent assistant turn of the history
(``Qwen2OmniPlugin.process_messages``, both ``mode="train"`` and ``mode="infer"``).  What
ROMA never learned is to *generate* it, which is what ``native=False`` records.

Four properties of the patched ``Qwen2_5OmniThinker`` dictate the prompt built below, and
none of them are negotiable.

**One second per chunk, one grid step per second.**  The patched
``get_interleaved_rope_index`` scans for a ``<|vision_bos|>`` immediately followed by
``<|audio_bos|>`` and, for each such block, advances its cursor by a hard-coded *one*
temporal grid step plus ``position_id_per_seconds`` (25) audio tokens.  A chunk must
therefore contribute exactly one temporal grid step, which -- ``temporal_patch_size`` being
2 -- means exactly two frames.  ``required_frames_per_chunk`` exists so the runner cannot
get this wrong.

**Uniform frame geometry.**  That same function reads ``video_grid_thw[i, 1:3]`` with ``i``
indexing the *batch*, not the chunk, so every chunk of a row has to resize to one height
and width.  Frames of a row come from one source video, so this holds in practice; it is
checked rather than assumed.

**Audio is always present.**  The rope walker only recognises a chunk when the vision and
audio openers are adjacent, and ROMA's own pipeline zero-pads the audio track to the full
video length (``_regularize_audios``).  A run configured without audio therefore feeds
*silent* audio rather than no audio: identical token layout, identical positions, no
acoustic evidence.  That is what makes the with/without-audio pair differ in exactly one
thing, which is the entire point of running both.

**The context is 32768 tokens**, this checkpoint's architectural ceiling rather than a
configured limit.  The suite's 30-chunk context window keeps every row well inside it -- a
poll's prompt saturates around 23k tokens no matter how long the recording is -- so the guard
below is insurance rather than a coverage limit: it refuses loudly rather than silently
truncating a row's history.  See ``docs/baselines.md`` for the arithmetic.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import SILENCE
from ..conversation import Request
from ..media import AUDIO, IMAGE, Capabilities
from .base import LONGEST_EDGE, SHORTEST_EDGE, Dialect, GenerationConfig, ModelError
from .local import LocalModel, trim_completion

DEFAULT_CHECKPOINT = "/chatgpt_nas/dukaixuan.dkx/models/ROMA"

# 0.6 is the threshold ROMA's own proactive demo ships (``gradio/proactive_gradio.py``).
# It is a knob rather than a constant because the gate probability is recorded per turn, so
# a sweep can be replayed offline from one run's records.
DEFAULT_GATE_THRESHOLD = 0.6
# "mlp" reads the two-layer head (``gate_head_pro_fc1`` -> act -> ``gate_head_pro_fc2``),
# "gate" reads the single linear ``gate_head``.  Both sets of weights are in the checkpoint;
# ROMA's evaluation scripts default to the MLP.
DEFAULT_GATE_HEAD = "mlp"
# Layers whose hidden states the gate mixes, when the checkpoint does not name its own.
DEFAULT_GATE_LAYER_IDS = (-4, -3, -2, -1)

# Fixed by the checkpoint: ``position_id_per_seconds`` in thinker_config, and hard-coded a
# second time inside the patched rope walker.
AUDIO_TOKENS_PER_CHUNK = 25
# 16000 samples / hop_length 160 = 100 mel frames, which the audio tower reduces to the 25
# tokens above.  Chunking the waveform one second at a time is also what keeps the tower
# inside its 1500-position limit on a 45-second row.
SAMPLES_PER_SECOND = 16000
MEL_FRAMES_PER_SECOND = 100
FRAMES_PER_CHUNK = 2

# Verbatim ``default_system`` of ROMA's ``streaming_mix`` template
# (``src/llamafactory/data/template.py``), down to the four-space continuation indent and the
# trailing newline-and-indent.  ROMA's own scripts pass ``system=""``, and the template does
# ``system = system or self.default_system``, so this is the string the checkpoint was trained
# and evaluated under -- including its instruction to answer ``"alert"`` rather than a
# sentence.  ROMA's repository is a working tree rather than a release, so the template can
# change under this copy; re-diff it against the source after updating that clone.
ROMA_SYSTEM_PROMPT = (
    "You are an assistant that receives streaming multimodal inputs. You must continuously"
    " monitor the synchronized video and audio each second and respond at the appropriate"
    " time. Provide answers to the audio query based on the visual context at the right"
    " moment.\n"
    "    [narration time] 1. You should only predict a response when an event transition"
    " occurs, and at that moment summarize the previous event. After responding, you must not"
    " predict another response until the next event transition.\n"
    "    [narration content] 2. When a response is required, follow the instructions and give"
    " a concise summary of the most recent event that just occurred.\n"
    "    [proactive alert] 1. Only when the condition described in the user's audio"
    " instruction is actually satisfied in the current video/audio input are you allowed to"
    ' speak; otherwise you must output exactly "no".\n'
    "    [proactive alert] 2. When the condition is satisfied, if the instruction specifies an"
    " explicit response you must output that exact content, otherwise you must output exactly"
    ' "alert".\n'
    "    "
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_BOS = "<|vision_bos|>"
VISION_EOS = "<|vision_eos|>"
AUDIO_BOS = "<|audio_bos|>"
AUDIO_EOS = "<|audio_eos|>"
VIDEO_TOKEN = "<|VIDEO|>"
AUDIO_TOKEN = "<|AUDIO|>"

# ROMA's chunk turn is the media and nothing else -- no wrapper, no timestamp.  That is its
# own format rather than a stripped-down one: ``Qwen2OmniPlugin.process_messages`` appends
# ``{"role": "user", "content": media_content_this_chunk}`` per second, and the clock reaches
# the model positionally through the interleaved rope.  See ``_prompt`` for why writing the
# time in would be the departure here.  AURA and M4 land on the same wordless turn for their
# own reasons; only JoyAI and MOSS-VL print a clock, because only they write one.
#
# The opening query turn has no counterpart in that function either -- ROMA's own query
# arrives as audio, so its repository never spells one in text.  It is therefore the question
# and nothing else: borrowing the corpus's ``<query>``/``<stream_info>`` wrapper to fill the
# gap would show ROMA text no checkpoint of it has seen, and inventing a wrapper would be
# worse still.  What the audio path carries is the question; that is what gets written.
CHUNK_TURN = "{media}"
QUERY_TURN = "{text}"
# ROMA's silent turns are masked out of the LM loss during training, so it never learned to
# emit any silence string at all -- the decision lives entirely in the gate head, and
# ``native=False`` records that the silence on the output side is the harness's marker.  The
# string itself is ROMA's own: its preprocessing writes ``<|silence|>`` into every silent
# assistant turn of the history it reads back.  The system prompt is ROMA's, not the suite's,
# so the checkpoint is asked the way its own repository asks it.
DIALECT = Dialect(
    silence=SILENCE,
    system_prompt=ROMA_SYSTEM_PROMPT,
    chunk_turn=CHUNK_TURN,
    query_turn=QUERY_TURN,
    native=False,
)


# Sentinel so the class-level vision patch is installed exactly once per process.
_BLOCKWISE_VISION_PATCH_FLAG = "_proactive_bench_blockwise_sdpa"


def _blockwise_vision_sdpa_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    """Memory-scalable drop-in for ``Qwen2_5OmniVisionSdpaAttention.forward``.

    Without flash-attn the checkpoint's vision tower runs its sdpa attention class, which
    builds a dense ``[1, seq, seq]`` boolean block-diagonal mask from ``cu_seqlens`` and
    hands it to ``scaled_dot_product_attention``.  A dense mask disqualifies torch's flash
    and memory-efficient kernels, so the call falls to the math backend, which materialises
    a float32 ``[num_heads, seq, seq]`` score tensor -- 31 GiB for a single high-resolution
    multi-chunk poll -- and OOMs on an 80 GB card even though the model itself fits.

    Block-diagonal attention is exactly independent full attention within each ``cu_seqlens``
    block (per frame on the four full-attention layers, per 64-patch window on the rest), so
    this runs one *maskless* sdpa per block instead.  With no mask torch uses its built-in
    flash / mem-efficient kernel (O(seq) memory), which is the sdpa analogue of the
    ``flash_attn_varlen`` path the checkpoint takes when flash-attn is present -- numerically
    the same attention, just without the package.  Equal-length blocks are batched into one
    call so the many tiny windowed blocks stay a couple of kernel launches, not hundreds.
    """
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_rotary_pos_emb_vision,
    )

    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    q = q.transpose(0, 1)  # [heads, seq, head_dim]
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    bounds = cu_seqlens.tolist()
    starts_by_len: Dict[int, List[int]] = defaultdict(list)
    for i in range(1, len(bounds)):
        length = bounds[i] - bounds[i - 1]
        if length > 0:
            starts_by_len[length].append(bounds[i - 1])

    out = torch.empty_like(q)
    for length, starts in starts_by_len.items():
        starts_t = torch.as_tensor(starts, device=q.device, dtype=torch.long)
        offsets = torch.arange(length, device=q.device, dtype=torch.long)
        # Block-major index: start0, start0+1, ..., start1, ...  so a plain reshape splits
        # it back into [heads, n_blocks, length, head_dim] with no scatter arithmetic.
        index = (starts_t.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
        n = len(starts)
        qb = q.index_select(1, index).reshape(self.num_heads, n, length, -1).transpose(0, 1)
        kb = k.index_select(1, index).reshape(self.num_heads, n, length, -1).transpose(0, 1)
        vb = v.index_select(1, index).reshape(self.num_heads, n, length, -1).transpose(0, 1)
        ob = F.scaled_dot_product_attention(qb, kb, vb, dropout_p=0.0)  # [n, heads, length, hd]
        ob = ob.transpose(0, 1).reshape(self.num_heads, n * length, -1)
        out.index_copy_(1, index, ob)

    attn_output = out.transpose(0, 1).reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


def _install_blockwise_vision_sdpa() -> bool:
    """Swap the vision sdpa attention forward for the block-diagonal one, once.

    Returns True if the patch is in place (installed now or already), False if the sdpa
    attention class cannot be located -- in which case the caller stays on the stock path.
    """
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniVisionSdpaAttention,
        )
    except Exception:  # pragma: no cover - wrong transformers, nothing to patch
        return False
    if getattr(Qwen2_5OmniVisionSdpaAttention, _BLOCKWISE_VISION_PATCH_FLAG, False):
        return True
    Qwen2_5OmniVisionSdpaAttention.forward = _blockwise_vision_sdpa_forward
    setattr(Qwen2_5OmniVisionSdpaAttention, _BLOCKWISE_VISION_PATCH_FLAG, True)
    return True


class ROMA(LocalModel):
    """ROMA driven through its gate head, loaded in-process."""

    default_name = "roma"
    dialect = DIALECT
    has_audio_encoder = True
    # Read by ``run_eval.py``: two frames per chunk is a property of the model's rope, not a
    # preference, so the runner adopts it instead of letting the flag default win.
    required_frames_per_chunk = FRAMES_PER_CHUNK

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        audio: bool = True,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
        gate_head: str = DEFAULT_GATE_HEAD,
        shortest_edge: int = SHORTEST_EDGE,
        longest_edge: int = LONGEST_EDGE,
    ) -> None:
        if gate_head not in ("mlp", "gate"):
            raise ValueError(f"gate_head must be 'mlp' or 'gate', got {gate_head!r}")
        super().__init__(
            checkpoint=checkpoint,
            # Frames, never the mp4: ROMA's own video regulariser reads a single ``.mp4`` or
            # a list of image paths, and our rows carry one short mp4 per chunk, which it
            # would try to open as an image.  Frames also make the two-frames-per-chunk
            # requirement expressible at all.
            capabilities=Capabilities(video=False, image=True, audio=bool(audio)),
            generation=generation,
            display_name=display_name,
            device=device,
        )
        self.attn_implementation = attn_implementation
        self.attn_used = ""
        self.vision_blockwise_sdpa = False
        self.gate_threshold = float(gate_threshold)
        self.gate_head = gate_head
        self.shortest_edge = int(shortest_edge)
        self.longest_edge = int(longest_edge)
        self.context_limit = 0

    def _load(self) -> None:
        try:
            import torch
            from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
        except ImportError as exc:  # pragma: no cover - environment problem, not a bug
            raise ModelError(
                "the roma backend needs ROMA's patched transformers (the roma_patch branch "
                "of ROMA_github/transformers): stock transformers has no gate head and no "
                "get_interleaved_rope_index"
            ) from exc

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.checkpoint)
        # The checkpoint's own preprocessor allows up to 12845056 pixels, 21x this suite's
        # budget.  Held to its own config ROMA would be scored on a far sharper frame than
        # everyone else, so the shared budget is written onto the image processor and left
        # as the only resize constraint -- which is why ROMA wants ``--frame-max-side 0``.
        self.processor.image_processor.min_pixels = self.shortest_edge
        self.processor.image_processor.max_pixels = self.longest_edge
        self.processor.image_processor.size = {
            "shortest_edge": self.shortest_edge,
            "longest_edge": self.longest_edge,
        }

        # The patched transformers is pinned to 4.50.0.dev0 and its own torch, and flash-attn
        # is not guaranteed to have a wheel for that pair.  Falling back beats not running the
        # baseline, but it goes into the run record rather than being swallowed: sdpa is close
        # to flash attention, not identical to it.
        attn = self.attn_implementation
        if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            attn = "sdpa"
        self.attn_used = attn
        # On the sdpa fallback the vision tower would otherwise materialise a float32
        # [heads, seq, seq] score matrix (dense block-diagonal mask -> math backend) and OOM
        # on high-resolution multi-chunk polls; the block-wise maskless rewrite keeps it on
        # torch's built-in flash / mem-efficient kernel.  Nothing to do on the flash path,
        # whose vision tower uses flash_attn_varlen and never builds the mask.
        self.vision_blockwise_sdpa = attn == "sdpa" and _install_blockwise_vision_sdpa()
        self.model = Qwen2_5OmniModel.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn,
            trust_remote_code=True,
        )
        # Only the thinker is ever called: the gate lives on it, and speech output is not
        # part of this benchmark.  Dropping the talker frees its weights.
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.eval()
        self.thinker = self.model.thinker
        self.tokenizer = self.processor.tokenizer
        self.context_limit = int(self.model.config.thinker_config.text_config.max_position_embeddings)

        for attribute in ("gate_mixer", f"gate_head_pro_fc1" if self.gate_head == "mlp" else "gate_head"):
            if not hasattr(self.thinker, attribute):
                raise ModelError(
                    f"checkpoint at {self.checkpoint} has no thinker.{attribute}: this is not "
                    "a gate-trained ROMA checkpoint"
                )

    # -- prompt building ---------------------------------------------------------------

    def _plan(self, request: Request) -> Tuple[List[Tuple[str, List[object]]], List[Dict[str, Any]]]:
        """Split the prefix into turns and per-chunk media groups.

        A user turn's content arrives as a flat list of strings and media refs; consecutive
        refs (two frames, optionally a track) are one second of stream and become one chunk.
        """
        chunks: List[Dict[str, Any]] = []
        turns: List[Tuple[str, List[object]]] = []
        for turn in request.turns:
            if isinstance(turn.content, str):
                turns.append((turn.role, [turn.content]))
                continue
            items: List[object] = []
            frames: List[str] = []
            track: Optional[str] = None
            for piece in turn.content:
                if isinstance(piece, str):
                    if frames:
                        chunks.append({"frames": frames, "audio": track})
                        items.append(len(chunks) - 1)
                        frames, track = [], None
                    items.append(piece)
                elif piece.kind == IMAGE:
                    frames.append(piece.path)
                elif piece.kind == AUDIO:
                    track = piece.path
                else:
                    raise ModelError(f"{self.name} cannot consume <{piece.kind}> input")
            if frames:
                chunks.append({"frames": frames, "audio": track})
                items.append(len(chunks) - 1)
            turns.append((turn.role, items))
        return turns, chunks

    def _media_block(self, video_tokens: int) -> str:
        return (
            VISION_BOS
            + AUDIO_BOS
            + VIDEO_TOKEN * video_tokens
            + AUDIO_TOKEN * AUDIO_TOKENS_PER_CHUNK
            + AUDIO_EOS
            + VISION_EOS
        )

    def _prompt(self, turns: Sequence[Tuple[str, List[object]]], video_tokens: int) -> str:
        """Render ROMA's ``streaming_mix`` layout for the prefix.

        A chunk turn carries only the media block, with no ``<chunk>`` wrapper and no
        ``<stream_info>`` timestamps -- declared as ``CHUNK_TURN`` above, so the turn arrives
        here already holding nothing else.  This is **settled**, and by ROMA's own
        preprocessing rather than by an argument about rope.
        ``Qwen2OmniPlugin.process_messages`` builds ROMA's per-second stream itself, and the
        user turn it appends for each second is::

            {"role": "user", "content": media_content_this_chunk.strip()}

        where ``media_content_this_chunk`` is exactly the ``<|vision_bos|><|audio_bos|>`` ...
        ``<|audio_eos|><|vision_eos|>`` block.  There is no timestamp text anywhere in the
        stream ROMA was trained and evaluated on; the clock reaches it positionally, through
        the interleaved rope.  The earlier worry -- that an assistant turn which speaks shifts
        a later chunk's position away from its wall-clock second -- is not an argument for
        adding timestamps here, because ROMA's training data has that same property: answers
        are inserted as assistant turns at ``chunk_idx + 1`` by the same function.  Writing
        timestamps in would be the departure, not leaving them out.

        A silent assistant turn is *not* empty, which is the second thing that function
        settles: it writes the literal ``<|silence|>`` for every second with no scheduled
        answer, in both ``mode="train"`` and ``mode="infer"``.  Contributing nothing would
        also leave the turn unclosed, running one ``<|im_start|>assistant`` straight into the
        next ``<|im_start|>user``.
        """
        out: List[str] = []
        for role, items in turns:
            media = [item for item in items if isinstance(item, int)]
            if media:
                body = "".join(self._media_block(video_tokens) for _ in media)
            else:
                body = "".join(item for item in items if isinstance(item, str))
            if role == "system":
                out.append(f"{IM_START}system\n{body}{IM_END}\n")
            elif role == "user":
                # ROMA's format_query ends on the assistant opener, so the rendered prefix
                # already stops exactly where the gate is read.
                out.append(f"{IM_START}user\n{body}{IM_END}\n{IM_START}assistant\n")
            else:
                spoken = body.strip()
                if spoken and self.dialect.spoke(spoken):
                    out.append(f"{spoken}{IM_END}\n")
                else:
                    out.append(f"{SILENCE}{IM_END}\n")
        return "".join(out)

    # -- media encoding ----------------------------------------------------------------

    def _waveform(self, path: Optional[str]):
        """One second of audio as float32 samples; silence when the run has no track."""
        import numpy as np

        if not path:
            return np.zeros(SAMPLES_PER_SECOND, dtype=np.float32)
        try:
            import librosa

            wave, _ = librosa.load(path, sr=SAMPLES_PER_SECOND, mono=True)
        except ImportError:
            # The workshop writes 16 kHz mono PCM wav, so the standard library is enough
            # when librosa is not installed.
            import wave as wave_module

            if not path.lower().endswith(".wav"):
                raise ModelError(f"librosa is required to read {path}")
            with wave_module.open(path, "rb") as handle:
                frames = handle.readframes(handle.getnframes())
            wave = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if wave.size == 0:
            return np.zeros(SAMPLES_PER_SECOND, dtype=np.float32)
        return wave

    def _encode_media(self, chunks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
        """Build the video and audio tensors, one grid step and 25 audio tokens per chunk."""
        import torch
        from PIL import Image

        if not chunks:
            raise ModelError(f"{self.name} was asked for a turn with no stream chunks")

        pixels: List[Any] = []
        features: List[Any] = []
        masks: List[Any] = []
        geometry: Optional[List[int]] = None

        for index, chunk in enumerate(chunks):
            frames = chunk["frames"]
            if len(frames) != FRAMES_PER_CHUNK:
                raise ModelError(
                    f"{self.name} needs exactly {FRAMES_PER_CHUNK} frames per chunk "
                    f"(temporal_patch_size), chunk {index} has {len(frames)}: run with "
                    f"--frames-per-chunk {FRAMES_PER_CHUNK}"
                )
            images = [Image.open(path).convert("RGB") for path in frames]
            try:
                encoded = self.processor.image_processor(
                    images=None, videos=[images], return_tensors="pt"
                )
            finally:
                for image in images:
                    image.close()
            grid = [int(value) for value in encoded["video_grid_thw"][0].tolist()]
            if grid[0] != 1:
                raise ModelError(
                    f"{self.name} chunk {index} produced {grid[0]} temporal grid steps; the "
                    "patched rope walker assumes exactly one per second"
                )
            if geometry is None:
                geometry = grid
            elif grid[1:] != geometry[1:]:
                raise ModelError(
                    f"{self.name} chunk {index} resized to {grid[1]}x{grid[2]} but chunk 0 "
                    f"to {geometry[1]}x{geometry[2]}; the patched rope walker reads one "
                    "geometry for the whole sequence"
                )
            pixels.append(encoded["pixel_values_videos"])

            audio = self.processor.feature_extractor(
                [self._waveform(chunk["audio"])],
                sampling_rate=SAMPLES_PER_SECOND,
                return_attention_mask=True,
                padding="max_length",
                max_length=SAMPLES_PER_SECOND,
                return_tensors="pt",
            )
            if int(audio["input_features"].shape[-1]) != MEL_FRAMES_PER_SECOND:
                raise ModelError(
                    f"{self.name} chunk {index} produced "
                    f"{int(audio['input_features'].shape[-1])} mel frames, expected "
                    f"{MEL_FRAMES_PER_SECOND}"
                )
            features.append(audio["input_features"])
            masks.append(audio["attention_mask"])

        merge = int(self.processor.image_processor.merge_size)
        video_tokens = (geometry[1] * geometry[2]) // (merge * merge)
        inputs = {
            "pixel_values_videos": torch.cat(pixels, dim=0),
            # One video of N temporal steps: the vision tower attends within each step, so
            # this is identical to N per-chunk encodes while keeping the chunk boundaries.
            "video_grid_thw": torch.tensor([[len(chunks), geometry[1], geometry[2]]]),
            # One row per second, so the audio tower encodes each second on its own and
            # emits exactly 25 tokens for it.
            "input_features": torch.cat(features, dim=0),
            "feature_attention_mask": torch.cat(masks, dim=0),
        }
        return inputs, video_tokens

    # -- inference ---------------------------------------------------------------------

    def _gate_probability(self, hidden_states) -> float:
        import torch

        layers = hidden_states
        if isinstance(hidden_states, tuple) and isinstance(hidden_states[1], (list, tuple)):
            # The patched thinker returns (embeds_to_talker, per-layer states).
            layers = hidden_states[1]
        total = len(layers)
        weights = self.thinker.gate_mixer.weights()
        layer_ids = getattr(self.thinker, "gate_layer_ids", list(DEFAULT_GATE_LAYER_IDS))
        mixed = None
        for weight, layer_id in zip(weights, layer_ids):
            index = int(layer_id if layer_id >= 0 else total + layer_id)
            index = max(0, min(total - 1, index))
            # The gate always reads the last position, which the prefix has arranged to be
            # the assistant opener -- the very position being asked about.
            anchor = layers[index][:, -1:, :]
            mixed = anchor * weight if mixed is None else mixed + anchor * weight
        if self.gate_head == "mlp":
            logit = self.thinker.gate_head_pro_fc2(
                self.thinker.gate_head_pro_act(self.thinker.gate_head_pro_fc1(mixed))
            )
        else:
            logit = self.thinker.gate_head(mixed)
        return float(torch.sigmoid(logit.reshape(-1)[0].float()))

    def generate_with_diagnostics(self, request: Request) -> Tuple[str, Dict[str, Any]]:
        self.ensure_loaded()
        import torch

        turns, chunks = self._plan(request)
        media, video_tokens = self._encode_media(chunks)
        prompt = self._prompt(turns, video_tokens)
        # skip_special_tokens is irrelevant here but add_special_tokens is not: the prompt
        # already carries every marker the template needs.
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        budget = self.context_limit - self.generation.max_tokens
        if int(input_ids.shape[-1]) > budget:
            raise ModelError(
                f"{self.name} prompt is {int(input_ids.shape[-1])} tokens for {len(chunks)} "
                f"chunks ({video_tokens} video + {AUDIO_TOKENS_PER_CHUNK} audio tokens per "
                f"second) and does not leave room for {self.generation.max_tokens} new "
                f"tokens in a {self.context_limit}-token context, which is this "
                f"checkpoint's architectural ceiling. Lower --longest-edge (now "
                f"{self.longest_edge}); 524288 fits every row in the delivered suites"
            )

        device = self.thinker.device
        # The processors emit float32; the checkpoint is bfloat16, and its patch embedding is a
        # convolution, which refuses a dtype it does not share (unlike a Linear, it will not
        # promote).  ROMA's own script casts exactly these two tensors and leaves the integer
        # ones alone (``omnimmi_qvh.py``: ``.to(model.dtype)`` on ``pixel_values_videos`` and
        # ``input_features``, plain ``.to(device)`` on the masks and the grid).
        dtype = self.thinker.dtype
        inputs = {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in media.items()
        }
        inputs["input_ids"] = input_ids.to(device)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.inference_mode():
                # position_ids is deliberately not passed: the patched forward computes the
                # interleaved rope itself whenever positions are absent and the cache is
                # empty, and reproducing that arithmetic here would be a second place to
                # get it wrong.
                outputs = self.thinker(
                    **inputs,
                    use_audio_in_video=True,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                probability = self._gate_probability(outputs.hidden_states)
        except Exception as exc:  # noqa: BLE001 - one bad row must not end the run
            raise ModelError(f"{self.name} gate evaluation failed: {exc}") from exc

        diagnostics: Dict[str, Any] = {
            "gate_prob": probability,
            "gate_threshold": self.gate_threshold,
            "gate_head": self.gate_head,
            "gate_fired": probability > self.gate_threshold,
            "prompt_tokens": int(input_ids.shape[-1]),
            "chunks": len(chunks),
            "video_tokens_per_chunk": video_tokens,
            "audio_tokens_per_chunk": AUDIO_TOKENS_PER_CHUNK,
            "audio_fed": bool(self.capabilities.audio and request.use_audio),
        }
        if not diagnostics["gate_fired"]:
            # The gate is the answer.  ROMA never learned to emit the silence token, so the
            # harness's own marker stands in for "said nothing"; the probability that led
            # here is on the record either way.
            return SILENCE, diagnostics

        try:
            with torch.inference_mode():
                # A second prefill rather than a reuse of the probe's cache: generate needs
                # to see cache_position 0 to build the interleaved rope, and correctness on
                # the minority of turns that speak is worth more than the saved forward.
                output = self.thinker.generate(
                    **inputs,
                    use_audio_in_video=True,
                    max_new_tokens=self.generation.max_tokens,
                    do_sample=False,
                )
        except Exception as exc:  # noqa: BLE001 - one bad row must not end the run
            raise ModelError(f"{self.name} generation failed: {exc}") from exc

        sequences = output.sequences if hasattr(output, "sequences") else output
        completion = self.tokenizer.decode(
            sequences[0][int(inputs["input_ids"].shape[-1]) :], skip_special_tokens=False
        )
        return trim_completion(completion), diagnostics

    def generate(self, request: Request) -> str:
        return self.generate_with_diagnostics(request)[0]

    def probe(self) -> str:
        missing = [
            name
            for name in ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
            if not os.path.isfile(os.path.join(self.checkpoint, name))
        ]
        if missing:
            return f"{self.checkpoint} (incomplete: missing {', '.join(missing)})"
        return f"{self.checkpoint} (not loaded yet)"

    def close(self) -> None:
        # The thinker is a second reference to weights ``super().close()`` is about to drop;
        # leaving it bound would keep the whole model alive on the GPU.
        self.thinker = None
        super().close()

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({
            "vision_input": "frames",
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "attn_implementation": self.attn_used or self.attn_implementation,
            "attn_implementation_requested": self.attn_implementation,
            # True when the sdpa fallback runs the vision tower block by block (maskless) so
            # it never materialises the [heads, seq, seq] score matrix that OOMs otherwise.
            "vision_blockwise_sdpa": self.vision_blockwise_sdpa,
            "silence_signal": "gate_head",
            "native_silence_token": False,
            "gate_head": self.gate_head,
            "gate_threshold": self.gate_threshold,
            # False under --audio off.  The token layout is identical either way; only the
            # acoustic evidence changes, which is what makes that ablation clean.  Paired
            # with has_audio_encoder so an ablation run stays distinguishable from a model
            # that has no ears in the first place.
            "has_audio_encoder": self.has_audio_encoder,
            "audio_fed": self.capabilities.audio,
            # A run without audio still gets 25 audio tokens per second, of silence: the
            # patched rope walker only recognises a chunk whose vision and audio openers are
            # adjacent, so dropping the stream would change the geometry, not just the sound.
            "silent_audio_when_unfed": True,
            "shortest_edge": self.shortest_edge,
            "longest_edge": self.longest_edge,
            "at_shared_budget": self.longest_edge == LONGEST_EDGE,
            "context_limit": self.context_limit or 32768,
            "audio_tokens_per_chunk": AUDIO_TOKENS_PER_CHUNK,
        })
        return info
