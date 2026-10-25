"""MOSS-VL-Realtime baseline.

The closest baseline to our setting: it was trained on streaming frames with a silence
token of its own, so it needs no prompt scaffolding to express "say nothing" -- it emits
``<|silence|>`` natively (token 151671 in this checkpoint).  It is still vision-only.

The checkpoint ships its own ``modeling_moss_vl`` / ``processing_moss_vl``, loaded through
``trust_remote_code``.  We drive the ordinary processor-and-``generate`` path rather than
the repository's ``offline_generate`` queue API: the harness already owns the streaming
protocol (rebuild the prefix, ask for one turn, feed the answer forward), and a queue that
manages its own session state would duplicate and fight that.  Feeding frames also keeps us
off its ``video_processing_moss_vl`` decode path, which needs a torchcodec pinned older than
the one in this environment.

Its layout is its own, and it is not one chat turn per frame.  The sample input it ships for
a 30-frame stream is three messages -- system, user, and a *single* assistant turn holding
``<|silence|><|video|><|silence|><|video|>...``, 31 silence tokens against 30 frames
(``realtime_inference/data/aerial_xinjiang_tour_guide_30s.jsonl``).  Its realtime loop builds
exactly that: the prefill is system-plus-user with a generation prompt
(``modeling_moss_vl.py:2249-2258``), each frame's region is appended straight into the
still-open assistant turn (``3136-3142``), and a ``user`` turn is opened only when new *text*
arrives (``3118-3124``).  The whole stream -- every frame and every per-second decision --
therefore lives inside one assistant message, and ``_payload`` folds the harness's
composition into that shape.  The leading ``<|silence|>`` is not decoration: ``3125-3134``
inserts it by hand because "every assistant turn in training data starts with ``<|silence|>``
as the first token ... Confirmed with project team".

That fold is the one place this benchmark lets a baseline's prompt *structure* differ from
the others.  What is held equal is the multimodal content -- the same frames in the same
order at the same pixel budget -- not the number of chat turns used to deliver it.  The cost
of getting it wrong is not cosmetic, and this adapter had it wrong: asked in dozens of
alternating user/assistant turns instead, the model went nearly mute wherever its prefix held
only silence -- 0 of 30 on ``response_at_first_evidence`` (the retired subtask the first_event
pair was cut from), 0 of 83 on ``respond_within_window`` -- while still speaking on 14% of turns
in the profiles whose prefix carries real alert text.  That gap was the prompt being out of
distribution, not a finding about the checkpoint, and every number measured through it is void.

Borrowing ``generate`` costs us the one thing the queue API provided for free: the turn
boundary.  In the repository's own loop the silence token *is* the boundary --
``modeling_moss_vl.py:3613`` reads ``should_wait_for_new_input = next_tokens.item() ==
silence_token_id``, so the instant this model emits ``<|silence|>`` its turn is over and it
blocks for the next frame.  A speaking turn therefore *starts* with something other than
silence and runs until the model closes it with a silence token.  ``generate`` has no frame
queue to block on, so past that first silence token the model just keeps rolling, inventing
per-frame decisions for frames it was never shown; left unstopped it emits silence until it
hits ``max_tokens``.
That continuation is not an answer to this poll and must not be read as one: measured over
the 663 scored turns of a 151-row run, 101 of the 120 turns that looked like speech were
continuations behind a leading ``<|silence|>`` -- 84% of this model's apparent responses
were invented after it had already declined to speak.  So the silence token is passed to
``generate`` as a stop token, which restores the repository's boundary exactly and, as a
side effect, makes a silent poll cost one token instead of 256.

``Dialect.spoke`` then needs no special case: a silent poll decodes to the lone stop token and
a speaking one to ``<|response|>text<|silence|>``, and stripping the hidden markers plus the
silence string leaves nothing and something respectively.  That ``<|response|>`` is the
checkpoint's own doing -- it opened every speaking turn with it once the fold was right, and
because its loop appends sampled tokens verbatim (``3602``) the marker is part of what the
model reads back, so ``_decision`` replays remembered alerts the same way.
``max_tokens_per_second`` (12 in the repository's CLI) is not a second cap to copy --
``_real_time_sample`` spends it on ``time.sleep`` to pace wall-clock output, and pacing a
benchmark would only make it slower.

Placeholders differ from the dataset's: this checkpoint's chat template writes ``<|image|>``
and ``<|video|>``, which its processor then expands into pad tokens.  Folding the stream by
hand means writing those two strings into the text directly rather than handing the template
typed content items -- which is what its own realtime loop does as well (``3141``).

Feeding frames costs something that has to be paid back explicitly.  Its processor wraps a
*video* in ``<|vision_start|>...<|vision_end|>`` and stamps every frame inside with
``<|time_start|>1.0 seconds<|time_end|>`` (processing_moss_vl.py:636-640), and its realtime
loop builds byte-identical regions per frame (modeling_moss_vl.py:3141) -- the wrapper being
what its SFT label masking keys on.  The *image* path does neither: ``<|image|>`` becomes a
bare ``<|image_pad|>``, no region, no timestamp.  So the switch from chunk mp4s to frames
would have quietly deleted the one structure this model always sees around a frame.  The
dialect puts it back verbatim, which is only correct in frames mode -- in video mode the
processor still builds the region itself and wrapping it again would nest two.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from ..conversation import Request, turn_text
from ..media import AUDIO, IMAGE, VIDEO, Capabilities
from .base import LONGEST_EDGE, SHORTEST_EDGE, Dialect, GenerationConfig, ModelError
from .local import LocalModel, trim_completion

# Verbatim from the repository:
#   realtime_inference/run_online_inference.py:36-41  DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT
# The silence string happens to match the dataset's, so only the system prompt changes -- but
# it is declared here rather than inherited, because the match is a coincidence between two
# checkpoints and not something this adapter should rely on.
SILENCE_TOKEN = "<|silence|>"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)
# One frame as its own timestamped vision region, verbatim from both places the checkpoint
# builds one:
#   processing_moss_vl.py:636-640   f"{vision_start}{time_start}{t:.1f} seconds{time_end}{image}...{vision_end}"
#   modeling_moss_vl.py:3141        the same string, per frame, in the realtime loop
# The template writes ``<|image|>``, which the processor then turns into ``<|image_pad|>``.
CHUNK_TURN = "<|vision_start|><|time_start|>{start} seconds<|time_end|>{media}<|vision_end|>"
# Its user turn is the question inside the repo's own realtime wrapper, not the bare text:
# run_online_inference.py:42 DEFAULT_PROMPT prefixes the streaming instruction and a
# ``Question:`` label, and prefills that before any frame (prompt_frame_first defaults False,
# so ``initial_prompt`` is the wrapped prompt at 534).  The SFT sample it ships wraps its
# question the same way in Chinese (``{instruction}\n\n问题：{q}``, the byte-for-byte structure
# of DEFAULT_PROMPT) in ``realtime_inference/data/aerial_xinjiang_tour_guide_30s.jsonl``, so a
# bare question is a shape no checkpoint of it was trained or run on.  The realtime splice at
# modeling_moss_vl.py:3120-3124 adds no timestamp, so none is written here; dropping the row's
# ``query_time`` loses nothing, as it is 0.0 on every row of all delivery sets.
QUERY_TURN = (
    "As the video streams frame by frame, respond to this question whenever the visual "
    "evidence lets you answer or updates a previous answer; otherwise stay silent.\n\n"
    "Question: {text}"
)
# What its own renderer strips before asking whether anything was said, beyond the silence
# token: run_online_inference.py:47 HIDDEN_OUTPUT_TOKENS plus the two round markers removed
# alongside them at 404-410.  Only ``<|response|>`` has ever turned up in what this checkpoint
# actually emits; the rest are here because its renderer names them, so that a turn holding a
# bare marker and nothing else cannot score as speech.
HIDDEN_TOKENS = ("<|round_start|>", "<|round_end|>", "<|response|>", "<|assistant|>", "<go>")
# The marker it puts in front of a spoken turn -- 151672 in the vocabulary, right next to the
# silence token, and emitted on every speaking turn it produced once the stream was folded the
# way its own loop folds one (9 of 9, against 2 of 19 while the prompt was still malformed).
RESPONSE_TOKEN = "<|response|>"
# The token the template writes for one image, and the one it writes for a video; the
# processor expands both into pad tokens.  Written by hand because the stream is folded by
# hand -- see the module docstring.
IMAGE_TOKEN = "<|image|>"
VIDEO_TOKEN = "<|video|>"
DIALECT = Dialect(
    silence=SILENCE_TOKEN, system_prompt=SYSTEM_PROMPT, hidden=HIDDEN_TOKENS, native=True
)
DIALECT_FRAMES = replace(DIALECT, chunk_turn=CHUNK_TURN, query_turn=QUERY_TURN)

DEFAULT_CHECKPOINT = "/chatgpt_nas/dukaixuan.dkx/models/MOSS-VL-Realtime-0708"
DEFAULT_REPO = "/chatgpt_nas/dukaixuan.dkx/MOSS-VL"


class MossVL(LocalModel):
    """MOSS-VL-Realtime loaded in-process via ``trust_remote_code``."""

    default_name = "moss_vl"
    dialect = DIALECT

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        generation: Optional[GenerationConfig] = None,
        display_name: str = "",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        vision_input: str = "frames",
        shortest_edge: int = SHORTEST_EDGE,
        longest_edge: int = LONGEST_EDGE,
    ) -> None:
        if vision_input not in ("video", "frames"):
            raise ValueError(f"vision_input must be 'video' or 'frames', got {vision_input!r}")
        super().__init__(
            checkpoint=checkpoint,
            capabilities=Capabilities(video=vision_input == "video", image=True, audio=False),
            generation=generation,
            display_name=display_name,
            device=device,
        )
        self.vision_input = vision_input
        # Only frames need the vision region written by hand; see the module docstring.
        self.dialect = DIALECT_FRAMES if vision_input == "frames" else DIALECT
        self.attn_implementation = attn_implementation
        self.shortest_edge = int(shortest_edge)
        self.longest_edge = int(longest_edge)

    def processor_kwargs(self) -> Dict[str, Any]:
        """How the processor is constructed, separable from loading 22B of weights.

        This checkpoint's own preprocessor_config.json allows up to 16777216 pixels, i.e.
        27x the budget the served models are held to.  Left alone, MOSS-VL would be scored
        on a much sharper view of the same frame than everyone else.

        Lifted out of ``_load`` so the cap can be verified without a GPU.  The spelling is
        the part that has to be tested, not just the numbers: the served path was silently
        uncapped for a while because it used the one key that backend ignores, and a checker
        holding its own copy of the spelling would have agreed with the bug.

        Deliberately absent: ``fix_mistral_regex=True``.  Loading this checkpoint warns that
        its pre-tokenizer carries Mistral's buggy regex and that "this will lead to incorrect
        tokenization", which is alarming and, here, inert.  The flag does change the regex --
        checked, not assumed -- but over all 206 distinct strings in the 50 default rows both
        settings produce identical token ids, so it would move no score.  It stays off because
        MOSS-VL's own repo does not pass it, and the point of this adapter is to ask the model
        the way its own repo does.
        """
        return {
            "trust_remote_code": True,
            "frame_extract_num_threads": 1,
            "size": {"shortest_edge": self.shortest_edge, "longest_edge": self.longest_edge},
        }

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ModelError("torch and transformers are required for the moss_vl backend") from exc
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, **self.processor_kwargs())
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        self._stop_ids = self._turn_stop_ids()

    def _turn_stop_ids(self) -> List[int]:
        """The ids that end one poll: the checkpoint's own terminators plus ``<|silence|>``.

        Resolved through the tokenizer the way ``_real_time_sample`` does rather than from
        the id in the module docstring, so a re-tokenized checkpoint cannot silently move the
        token out from under us.  A missing token is fatal: without the stop the model runs
        past its turn boundary and the run scores its inventions (see the module docstring).
        """
        silence_id = self.processor.tokenizer.convert_tokens_to_ids(SILENCE_TOKEN)
        unk = getattr(self.processor.tokenizer, "unk_token_id", None)
        if silence_id is None or silence_id < 0 or silence_id == unk:
            raise ModelError(
                f"{self.name}: tokenizer does not know {SILENCE_TOKEN!r}, so a poll cannot be "
                "stopped at its turn boundary"
            )
        configured = getattr(self.model.generation_config, "eos_token_id", None)
        eos = configured if isinstance(configured, (list, tuple)) else [configured]
        return sorted({int(i) for i in (*eos, silence_id) if i is not None})

    # -- prompt building ---------------------------------------------------------------

    def _decision(self, text: str) -> str:
        """One replayed turn of history, written the way this model writes one.

        Inside a single assistant stream the silence token is the *only* thing separating one
        second's decision from the next frame's region -- the turn ends when it is emitted
        (``modeling_moss_vl.py:3613``), so history has to carry it.  A silent turn already is
        the bare token; a spoken one arrives from the dialect as plain text and is wrapped
        here into ``<|response|>text<|silence|>``.

        That wrapper is what the model's own context holds.  Its realtime loop appends every
        token it sampled straight back onto ``input_ids``
        (``modeling_moss_vl.py:3602``) -- ``HIDDEN_OUTPUT_TOKENS`` is stripped only when
        rendering for a human (``run_online_inference.py:404-410``), never from the running
        prefix.  So in any real session where it spoke at second 20 and is deciding at second
        30, the marker sits in front of what it said.  Its own decodes agree: every speaking
        turn produced under this layout opened with it.
        """
        body = text.strip()
        if not self.dialect.spoke(body):
            return SILENCE_TOKEN
        return RESPONSE_TOKEN + body + SILENCE_TOKEN

    def _stream(self, request: Request) -> Tuple[str, List[str], List[str]]:
        """The assistant stream: opening silence, then frame regions and decisions in order.

        Kept free of the tokenizer so the fold can be read and checked without 22B of weights
        on a GPU -- this is the part of the prompt that is this model's own shape, and it was
        wrong once already.
        """
        stream: List[str] = [SILENCE_TOKEN]
        images: List[str] = []
        videos: List[str] = []
        for turn in request.turns[3:]:
            if turn.role == "assistant":
                stream.append(self._decision(turn_text(turn)))
                continue
            if isinstance(turn.content, str):
                stream.append(turn.content)
                continue
            for piece in turn.content:
                if isinstance(piece, str):
                    stream.append(piece)
                elif piece.kind == IMAGE:
                    stream.append(IMAGE_TOKEN)
                    images.append(piece.path)
                elif piece.kind == VIDEO:
                    stream.append(VIDEO_TOKEN)
                    videos.append(piece.path)
                elif piece.kind == AUDIO:  # pragma: no cover - dropped by Capabilities
                    raise ModelError("moss_vl has no audio encoder")
        return "".join(stream), images, videos

    def _payload(self, request: Request) -> Tuple[str, List[str], List[str]]:
        """Fold one polled conversation into this checkpoint's realtime layout.

        The harness composes the stream the way most of these baselines read one: the query as
        its own turn, then a user turn per second with the assistant's decision between them.
        This checkpoint reads a different shape -- system, user, and one open assistant turn
        holding everything after that (see the module docstring) -- so the composition is
        folded here rather than in ``conversation.py``, which stays the same for every model.

        The prefill goes through ``apply_chat_template`` exactly as its session does
        (``modeling_moss_vl.py:2253``); the stream is concatenated, as its realtime loop
        concatenates it, instead of relying on ``continue_final_message`` to reopen a turn the
        template has already closed.

        The dataset's acknowledgement turn is dropped.  There is nowhere in this layout for it:
        the moment it occupies is the one the training format fills with the assistant turn's
        opening ``<|silence|>``.  It costs the model no information -- the query it answers is
        right above it, and the prefix's real content, the earlier alerts, is folded in below.
        """
        turns = request.turns
        if len(turns) < 4 or [turn.role for turn in turns[:3]] != ["system", "user", "assistant"]:
            raise ModelError(
                f"{self.name}: expected a conversation opening system/user/assistant and at "
                f"least one chunk turn, got {[turn.role for turn in turns[:4]]} -- the fold "
                "into this model's single assistant stream keys on that opening"
            )
        prefill = self.processor.apply_chat_template(
            [
                {"role": "system", "content": turn_text(turns[0])},
                {"role": "user", "content": turn_text(turns[1])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        stream, images, videos = self._stream(request)
        return prefill + stream, images, videos

    def generate(self, request: Request) -> str:
        self.ensure_loaded()
        import torch
        from PIL import Image

        text, image_paths, video_paths = self._payload(request)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        try:
            inputs = self.processor(
                text=[text],
                images=images or None,
                videos=video_paths or None,
                return_tensors="pt",
            )
            inputs = {
                key: (value.to(self.model.device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation.max_tokens,
                    do_sample=False,
                    eos_token_id=self._stop_ids,
                )
        except Exception as exc:  # noqa: BLE001 - one long row must not end the run
            raise ModelError(f"{self.name} generation failed: {exc}") from exc
        finally:
            for image in images:
                image.close()
        prompt_length = int(inputs["input_ids"].shape[-1])
        completion = self.processor.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=False
        )
        return trim_completion(completion)

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({
            "vision_input": self.vision_input,
            "attn_implementation": self.attn_implementation,
            "shortest_edge": self.shortest_edge,
            "longest_edge": self.longest_edge,
        })
        return info
