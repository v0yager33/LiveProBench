"""Row loading and prediction-file I/O.

Rows may arrive as JSONL or as the parquet mirror the generator writes alongside it;
parquet lands numpy scalars and arrays in the dicts, so everything is normalised through
:func:`plain` before any scoring code touches it.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Set

_APPEND_LOCKS: Dict[str, threading.Lock] = {}
_APPEND_LOCKS_GUARD = threading.Lock()


def plain(value: Any) -> Any:
    """Recursively turn numpy/pandas scalars and arrays into plain Python values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_rows(path: str, limit: int = 0) -> List[dict]:
    rows: List[dict] = []
    if path.endswith(".parquet"):
        import pandas as pd

        for row in pd.read_parquet(path).to_dict("records"):
            rows.append(plain(row))
            if limit and len(rows) >= limit:
                break
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def iter_records(path: str) -> Iterator[dict]:
    """Yield prediction records, skipping lines a killed run left half-written."""
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _APPEND_LOCKS_GUARD:
        lock = _APPEND_LOCKS.get(key)
        if lock is None:
            lock = _APPEND_LOCKS[key] = threading.Lock()
        return lock


def append_jsonl(path: str, obj: dict) -> None:
    """Append one record, serialising writers so concurrent rows cannot interleave."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False)
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def completed_sample_ids(path: str, retry_errors: bool = False) -> Set[str]:
    """Sample ids already in an output file.

    With ``retry_errors`` a row whose only record is a failure is treated as unfinished, so
    a resumed run retries it instead of inheriting a transport error forever.
    """
    done: Set[str] = set()
    failed: Set[str] = set()
    for record in iter_records(path):
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        sample_id = str(sample_id)
        if record.get("status") == "ok":
            done.add(sample_id)
        else:
            failed.add(sample_id)
    if retry_errors:
        return done
    return done | failed
