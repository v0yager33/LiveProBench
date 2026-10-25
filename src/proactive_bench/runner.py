"""Run orchestration: concurrency, resume and per-row fault isolation.

A run is a long-lived thing -- the full suite is tens of thousands of requests, each row
costing one request per polled turn -- so the runner is built around the assumption that it
will be interrupted and restarted:

* every finished row is appended to the output file immediately, so a kill loses at most
  the rows in flight;
* ``--resume`` skips sample ids the output already carries, and without it the previous
  attempt's file is set aside rather than appended to, so a transcript never holds two
  records for the same row;
* a row that fails is recorded with ``status != "ok"`` and the run continues, because the
  usual causes (a row with more video parts than the server's per-prompt limit, a prefix
  that outgrows the context window) are properties of that row alone.

Concurrency is capped by the model's own ``max_concurrency``: the resident server accepts
several rows at once, a locally loaded checkpoint must stay serial.
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .annotation import Annotation, from_row
from .dataset import append_jsonl, completed_sample_ids, iter_records, load_rows
from .scoring import Evaluator
from .suite import InputSpec


@dataclass
class RunConfig:
    output: str
    workers: int = 4
    limit: int = 0
    resume: bool = False
    retry_errors: bool = False
    progress: bool = True


@dataclass
class RunStats:
    total: int = 0
    scored: int = 0
    failed: int = 0
    skipped: int = 0
    started: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows_total": self.total,
            "rows_scored": self.scored,
            "rows_failed": self.failed,
            "rows_skipped": self.skipped,
            "wall_sec": round(time.time() - self.started, 1),
        }


def gather_annotations(specs: Sequence[InputSpec], limit: int = 0) -> List[Annotation]:
    """Load and validate every input file, applying ``limit`` per file rather than per run.

    Per file is what makes ``--limit`` useful for a smoke run: a single global cap would
    spend the whole budget on the first subtask and never touch the others.

    Rows are parsed into annotations here, before any request is sent, so a file that has
    drifted from the schema fails while it is still one identifiable row rather than as a
    puzzling media error thousands of polls into the run.  ``subtask`` is defaulted from the
    file the row came from: published rows carry it, a hand-trimmed debugging file may not,
    and the summary buckets by that field.
    """
    annotations: List[Annotation] = []
    for spec in specs:
        for row in load_rows(spec.path, limit=limit):
            if spec.subtask:
                row.setdefault("subtask", spec.subtask)
            annotations.append(from_row(row))
    return annotations


def collect_records(path: str) -> List[dict]:
    """All records for the summary, keeping the newest record per sample id.

    A resumed run that retried failures leaves both the old failure and the new success in
    the file; the later line is the one that counts.
    """
    latest: Dict[str, dict] = {}
    order: List[str] = []
    for record in iter_records(path):
        key = str(record.get("sample_id") or f"__anonymous_{len(order)}")
        if key not in latest:
            order.append(key)
        latest[key] = record
    return [latest[key] for key in order]


def _progress_bar(total: int, enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="row", dynamic_ncols=True)


class Runner:
    """Evaluates a set of rows against one model."""

    def __init__(self, evaluator: Evaluator, config: RunConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self.stats = RunStats()
        self._lock = threading.Lock()

    def _error_record(self, annotation: Annotation, exc: BaseException) -> Dict[str, Any]:
        return {
            "sample_id": annotation.sample_id,
            "plan_id": annotation.plan_id,
            "video_id": annotation.video.video_id,
            "subtask": annotation.subtask,
            "model": self.evaluator.model.name,
            "status": f"error:{type(exc).__name__}",
            "error": str(exc)[:600],
            "pass": False,
            "pass_strict": False,
            "content_ok": None,
            "predictions": [],
        }

    def _truncate_output(self) -> None:
        """Empty the predictions file, keeping the previous attempt beside it.

        The old lines are moved to ``<name>.superseded`` rather than dropped: a run is
        expensive enough that overwriting one on a mistyped ``--run-name`` should be
        recoverable, and the copy is one file per name so it cannot accumulate.
        """
        path = self.config.output
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        os.replace(path, path + ".superseded")
        print(f"note       : previous {os.path.basename(path)} moved to"
              f" {os.path.basename(path)}.superseded (no --resume)", file=sys.stderr)

    def _score(self, annotation: Annotation) -> Dict[str, Any]:
        try:
            return self.evaluator.evaluate(annotation)
        except Exception as exc:  # noqa: BLE001 - one row must not end the run
            return self._error_record(annotation, exc)

    def run(self, annotations: Sequence[Annotation]) -> List[dict]:
        pending = list(annotations)
        self.stats.total = len(pending)
        if self.config.resume:
            done = completed_sample_ids(self.config.output, retry_errors=self.config.retry_errors)
            if done:
                before = len(pending)
                pending = [item for item in pending if item.sample_id not in done]
                self.stats.skipped = before - len(pending)
        else:
            # Without --resume this is a fresh run, so the file must start empty. Appending
            # instead would leave the previous attempt's rows underneath the new ones: the
            # summary would still be right, because ``collect_records`` keeps the last record
            # per sample id, but the transcript would carry two contradictory lines for the
            # same row and anything reading it directly would have to know that rule to be
            # correct. Truncating is the same data loss the run was already asking for.
            self._truncate_output()

        workers = max(1, min(self.config.workers, self.evaluator.model.max_concurrency))
        bar = _progress_bar(len(pending), self.config.progress)

        def finish(record: Dict[str, Any]) -> None:
            with self._lock:
                if str(record.get("status")) == "ok":
                    self.stats.scored += 1
                else:
                    self.stats.failed += 1
                append_jsonl(self.config.output, record)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(ok=self.stats.scored, err=self.stats.failed, refresh=False)

        try:
            if workers == 1:
                for annotation in pending:
                    finish(self._score(annotation))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(self._score, item) for item in pending]
                    for future in concurrent.futures.as_completed(futures):
                        finish(future.result())
        except KeyboardInterrupt:
            # Everything finished so far is already on disk; --resume picks up from here.
            print("\ninterrupted; finished rows are on disk, rerun with --resume", file=sys.stderr)
        finally:
            if bar is not None:
                bar.close()
        return collect_records(self.config.output)
