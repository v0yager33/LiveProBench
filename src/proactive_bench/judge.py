"""Content judge: the ``content_ok`` side metric.

``pass`` only asks whether the model opened its mouth.  A reply like "nothing has happened
yet, I will keep watching" is speech and would otherwise take a free pass on a speak
target, so every speak target that did speak is additionally shown to a text-only judge.

No media is sent.  The dataset's reference alert already establishes what happened, so the
only open question is what the candidate reply claims -- and a text-only judge is both
cheaper and more reproducible than a second video pass.

The judge prompt and LLM client are reused from ``data_generation`` rather than copied: the
prompt file is the single source of truth for what counts as a valid alert, and a divergent
copy here would silently change the side metric.  Everything is imported lazily so a plain
pass-rate run needs neither that package nor any judge credential.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .annotation import Annotation, Target
from .dataset import plain

# The generator lives beside the harness; both paths are needed because the judge helpers
# import ``data_generation_pipeline`` from the project root.
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_GENERATION_DIR = os.path.join(BENCH_ROOT, "data_generation")
PROJECT_ROOT = "/chatgpt_nas/dukaixuan.dkx/LiveChatOmni"
PROMPT_NAME = "response_content_judge_system"

_PROMPT_CACHE: Dict[str, str] = {}
_IMPORT_LOCK = threading.Lock()


def _ensure_importable() -> None:
    with _IMPORT_LOCK:
        for path in (DATA_GENERATION_DIR, PROJECT_ROOT):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)


def load_prompt(name: str = PROMPT_NAME) -> str:
    if name not in _PROMPT_CACHE:
        path = os.path.join(DATA_GENERATION_DIR, "prompts", f"{name}.md")
        with open(path, encoding="utf-8") as handle:
            _PROMPT_CACHE[name] = handle.read().rstrip("\n")
    return _PROMPT_CACHE[name]


def target_event_text(meta: Dict[str, Any], chunk_idx: int) -> str:
    """Describe the occurrence a speak target is supposed to report."""
    for event in plain(meta.get("response_events")) or []:
        if isinstance(event, dict) and int(event.get("evidence_chunk", -1)) == chunk_idx:
            parts = [
                str(event.get("event_description") or ""),
                str(event.get("first_evidence_description") or ""),
            ]
            return " ".join(part for part in parts if part).strip()
    for key in ("first_evidence_description", "activity_description", "event_summary"):
        text = str(meta.get(key) or "").strip()
        if text:
            return text
    return ""


@dataclass
class JudgeConfig:
    provider: str = "gemini"
    model: str = ""
    slot_root: str = ""
    global_slots: int = 0


class ContentJudge:
    """A text-only verdict on whether a reply reports the awaited occurrence."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        _ensure_importable()
        from build_eval_proactive import resolve_eval_judge_config

        self.llm_config = resolve_eval_judge_config(config.provider, config.model)

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": getattr(self.llm_config, "model", self.config.model),
            "prompt": PROMPT_NAME,
        }

    def _ask(self, request: Dict[str, str]) -> str:
        from data_generation_pipeline.core.io_utils import global_slot
        from data_generation_pipeline.core.llm_client import build_llm_client

        messages = [
            {"role": "system", "content": load_prompt()},
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(request, ensure_ascii=False)}],
            },
        ]
        client = build_llm_client(self.llm_config)
        try:
            if self.config.slot_root and self.config.global_slots:
                with global_slot(self.config.slot_root, self.config.global_slots):
                    return client.chat_messages(messages) or ""
            return client.chat_messages(messages) or ""
        finally:
            client.close()

    def judge(self, annotation: Annotation, target: Target, response: str) -> Dict[str, Any]:
        """Return the ``content_ok`` fields for one spoken target turn.

        A judge that cannot be reached or whose answer cannot be parsed yields
        ``content_ok = None``, which keeps ``pass_strict`` equal to ``pass`` for that row.
        An unavailable judge must not be able to fail a model.
        """
        from data_generation_pipeline.core.prompts import extract_json_object

        # The generator's own rationales and event descriptions, carried through the
        # conversion untouched. This is the only consumer of that block.
        meta = annotation.meta
        request = {
            "user_alert_request": str(meta.get("instruction") or ""),
            "occurrence_that_just_happened": target_event_text(meta, target.chunk_idx),
            "reference_alert": str(target.expected or ""),
            "candidate_reply": str(response),
        }
        try:
            text = self._ask(request)
        except Exception as exc:  # noqa: BLE001 - the side metric never fails a run
            return {"content_ok": None, "content_judge_reason": f"judge error: {exc}"[:300]}
        obj = extract_json_object(text)
        if not obj:
            return {
                "content_ok": None,
                "content_judge_reason": "unparsed",
                "content_judge_raw": text[:400],
            }
        verdict = obj.get("confirms_the_event_happened")
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() in {"true", "yes", "1"}
        return {
            "content_ok": bool(verdict),
            "content_judge_reason": str(obj.get("reason") or "")[:300],
        }


def build_judge(enabled: bool, provider: str, model: str = "", slot_root: str = "",
                global_slots: int = 0) -> Optional[ContentJudge]:
    if not enabled:
        return None
    return ContentJudge(
        JudgeConfig(provider=provider, model=model, slot_root=slot_root, global_slots=global_slots)
    )
