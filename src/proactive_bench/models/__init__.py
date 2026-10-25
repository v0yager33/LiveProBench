"""Model registry.

One place knows which backends exist, what their defaults are, and which command-line
flags configure them.  Adding a baseline means adding an adapter module and one entry in
:data:`REGISTRY` -- the runner, the scorer and the CLI stay untouched.

Adapters are imported lazily by their factory, so a run against the HTTP endpoints never
imports torch and a run against a local checkpoint never needs the openai package.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import LONGEST_EDGE, SHORTEST_EDGE, GenerationConfig, ModelError, ProactiveModel


@dataclass
class ModelOptions:
    """Everything the CLI can say about how to build a model."""

    name: str
    # HTTP backends
    base_url: str = ""
    model_id: str = ""
    api_key: str = "EMPTY"
    workers: int = 0
    mm_processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    fps: float = 0.0
    shortest_edge: int = 0
    longest_edge: int = 0
    # Local backends
    checkpoint: str = ""
    repo: str = ""
    device: str = "cuda"
    attn_implementation: str = "flash_attention_2"
    # Both
    # Frames plus a per-chunk audio track is the layout the dataset is migrating to and the
    # form every model is scored on: the chunk mp4 is a transport detail of the current
    # files, not part of the task.  'video' remains available for a server-side comparison.
    vision_input: str = "frames"
    # Honoured only by the adapters that have an audio encoder (LiveFlow_omni, roma); for
    # the vision-only baselines audio is already absent and this changes nothing.
    audio: bool = True
    # ROMA only: its speak/silence decision is a gate probability, not text.
    gate_threshold: float = 0.0
    gate_head: str = ""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    display_name: str = ""


Factory = Callable[[ModelOptions], ProactiveModel]


def _LiveFlow_omni(options: ModelOptions) -> ProactiveModel:
    from .LiveFlow_omni import (
        DEFAULT_BASE_URL,
        DEFAULT_FPS,
        DEFAULT_LONGEST_EDGE,
        DEFAULT_MODEL_ID,
        DEFAULT_SHORTEST_EDGE,
        LiveChatOmni,
    )

    return LiveChatOmni(
        model_id=options.model_id or DEFAULT_MODEL_ID,
        base_url=options.base_url or DEFAULT_BASE_URL,
        generation=options.generation,
        display_name=options.display_name or options.name,
        workers=options.workers or 4,
        audio=options.audio,
        vision_input=options.vision_input,
        fps=options.fps or DEFAULT_FPS,
        shortest_edge=options.shortest_edge or DEFAULT_SHORTEST_EDGE,
        longest_edge=options.longest_edge or DEFAULT_LONGEST_EDGE,
    )


def _aura(options: ModelOptions) -> ProactiveModel:
    from .aura import AURA, DEFAULT_CHECKPOINT, DEFAULT_PORT

    return AURA(
        model_id=options.model_id or options.checkpoint or DEFAULT_CHECKPOINT,
        base_url=options.base_url or f"http://localhost:{DEFAULT_PORT}/v1",
        generation=options.generation,
        display_name=options.display_name or options.name,
        workers=options.workers or 4,
        vision_input=options.vision_input,
        mm_processor_kwargs=options.mm_processor_kwargs,
    )


def _joyai(options: ModelOptions) -> ProactiveModel:
    from .joyai import DEFAULT_CHECKPOINT, DEFAULT_PORT, JoyAI

    return JoyAI(
        model_id=options.model_id or options.checkpoint or DEFAULT_CHECKPOINT,
        base_url=options.base_url or f"http://localhost:{DEFAULT_PORT}/v1",
        generation=options.generation,
        display_name=options.display_name or options.name,
        workers=options.workers or 4,
        vision_input=options.vision_input,
        mm_processor_kwargs=options.mm_processor_kwargs,
    )


def _moss_vl(options: ModelOptions) -> ProactiveModel:
    from .moss_vl import DEFAULT_CHECKPOINT, MossVL

    return MossVL(
        checkpoint=options.checkpoint or DEFAULT_CHECKPOINT,
        generation=options.generation,
        display_name=options.display_name or options.name,
        device=options.device,
        attn_implementation=options.attn_implementation,
        vision_input=options.vision_input,
        shortest_edge=options.shortest_edge or SHORTEST_EDGE,
        longest_edge=options.longest_edge or LONGEST_EDGE,
    )


def _m4(options: ModelOptions) -> ProactiveModel:
    from .m4 import DEFAULT_CHECKPOINT, DEFAULT_REPO, M4

    return M4(
        checkpoint=options.checkpoint or DEFAULT_CHECKPOINT,
        repo=options.repo or DEFAULT_REPO,
        generation=options.generation,
        display_name=options.display_name or options.name,
        device=options.device,
        attn_implementation=options.attn_implementation,
    )


def _roma(options: ModelOptions) -> ProactiveModel:
    from .roma import DEFAULT_CHECKPOINT, DEFAULT_GATE_HEAD, DEFAULT_GATE_THRESHOLD, ROMA

    return ROMA(
        checkpoint=options.checkpoint or DEFAULT_CHECKPOINT,
        generation=options.generation,
        display_name=options.display_name or options.name,
        device=options.device,
        attn_implementation=options.attn_implementation,
        audio=options.audio,
        gate_threshold=options.gate_threshold or DEFAULT_GATE_THRESHOLD,
        gate_head=options.gate_head or DEFAULT_GATE_HEAD,
        shortest_edge=options.shortest_edge or SHORTEST_EDGE,
        longest_edge=options.longest_edge or LONGEST_EDGE,
    )


REGISTRY: Dict[str, Factory] = {
    "LiveFlow_omni": _LiveFlow_omni,
    "aura": _aura,
    "joyai": _joyai,
    "moss_vl": _moss_vl,
    "m4": _m4,
    "roma": _roma,
}

# Names used in the papers and repositories, mapped onto the registry keys.
ALIASES: Dict[str, str] = {
    "ours": "LiveFlow_omni",
    "omni": "LiveFlow_omni",
    "qwen3_omni_opsd": "LiveFlow_omni",
    "joyai_vl_interaction": "joyai",
    "joyai-vl-interaction": "joyai",
    "moss": "moss_vl",
    "moss_vl_realtime": "moss_vl",
    "moss-vl-realtime": "moss_vl",
}

# Backends that hold a GPU in this process and must not be run concurrently.
LOCAL_BACKENDS = frozenset({"moss_vl", "m4", "roma"})


def canonical_name(name: str) -> str:
    key = str(name).strip().replace("-", "_").lower()
    key = ALIASES.get(key, key)
    if key not in REGISTRY:
        raise ModelError(
            f"unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    return key


def available_models() -> List[str]:
    return sorted(REGISTRY)


def build_model(options: ModelOptions) -> ProactiveModel:
    key = canonical_name(options.name)
    resolved = ModelOptions(**{**options.__dict__, "name": key})
    if key in LOCAL_BACKENDS:
        # An in-process checkpoint has one set of weights and one CUDA stream; letting the
        # runner fan out would only interleave prompts through the same GPU.
        resolved.workers = 1
    return REGISTRY[key](resolved)


# -- command line ---------------------------------------------------------------------


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--model", required=True,
        help="Model to evaluate: " + ", ".join(sorted(REGISTRY)) + " (aliases accepted).",
    )
    group.add_argument("--model-name", default="",
                       help="Label recorded in results; defaults to --model.")
    group.add_argument("--base-url", default="",
                       help="OpenAI-compatible endpoint; defaults to the backend's own port.")
    group.add_argument("--model-id", default="",
                       help="Served model id. Must match the path the server was started with.")
    group.add_argument("--api-key", default="EMPTY")
    group.add_argument("--checkpoint", default="",
                       help="Weights directory for a local backend.")
    group.add_argument("--repo", default="",
                       help="Baseline repository to import (M4 only).")
    group.add_argument("--device", default="cuda")
    group.add_argument("--attn-implementation", default="flash_attention_2")
    group.add_argument(
        "--vision-input", choices=("video", "frames"), default="frames",
        help="Feed each chunk as extracted frames plus its audio track (the default, and "
             "the layout the dataset is migrating to), or as the chunk's mp4 file. Every "
             "model is scored on 'frames' so that the visual and acoustic content is "
             "identical across the suite regardless of what each server can decode.",
    )
    group.add_argument("--fps", type=float, default=0.0,
                       help="Frames per second requested from the server's video processor.")
    group.add_argument("--shortest-edge", type=int, default=0)
    group.add_argument("--longest-edge", type=int, default=0)
    group.add_argument("--mm-processor-kwargs", default="",
                       help="JSON object forwarded as extra_body.mm_processor_kwargs.")
    group.add_argument(
        "--audio", choices=("on", "off"), default="on",
        help="Feed the stream's audio track to a model that can hear it. 'off' runs the "
             "same rows vision-only, which is the ablation for LiveFlow_omni and roma; the "
             "vision-only baselines ignore it. An 'off' run is filed under its own "
             "'_no_audio' results directory so the two never merge under --resume.",
    )
    group.add_argument("--gate-threshold", type=float, default=0.0,
                       help="ROMA only: speak when the gate probability exceeds this "
                            "(default 0.6, the value ROMA's own demo ships). Every turn's "
                            "raw probability is recorded, so a sweep needs no re-run.")
    group.add_argument("--gate-head", choices=("mlp", "gate"), default="",
                       help="ROMA only: which of the checkpoint's two gate heads to read "
                            "(default mlp, as in ROMA's evaluation scripts).")

    group = parser.add_argument_group("generation")
    group.add_argument("--max-tokens", type=int, default=256)
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument("--retries", type=int, default=3)
    group.add_argument("--timeout", type=float, default=600.0)


def options_from_args(args: argparse.Namespace) -> ModelOptions:
    from .openai_chat import parse_mm_processor_kwargs

    return ModelOptions(
        name=args.model,
        base_url=args.base_url,
        model_id=args.model_id,
        api_key=args.api_key,
        workers=getattr(args, "workers", 0) or 0,
        mm_processor_kwargs=parse_mm_processor_kwargs(args.mm_processor_kwargs),
        fps=args.fps,
        shortest_edge=args.shortest_edge,
        longest_edge=args.longest_edge,
        checkpoint=args.checkpoint,
        repo=args.repo,
        device=args.device,
        attn_implementation=args.attn_implementation,
        vision_input=args.vision_input,
        audio=getattr(args, "audio", "on") != "off",
        gate_threshold=getattr(args, "gate_threshold", 0.0) or 0.0,
        gate_head=getattr(args, "gate_head", "") or "",
        generation=GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retries=args.retries,
            timeout=args.timeout,
        ),
        display_name=args.model_name or canonical_name(args.model),
    )


def model_from_args(args: argparse.Namespace) -> ProactiveModel:
    return build_model(options_from_args(args))


__all__ = [
    "ALIASES",
    "GenerationConfig",
    "LOCAL_BACKENDS",
    "ModelError",
    "ModelOptions",
    "ProactiveModel",
    "REGISTRY",
    "add_model_arguments",
    "available_models",
    "build_model",
    "canonical_name",
    "model_from_args",
    "options_from_args",
]
