#!/usr/bin/env python3
"""Re-derive a summary from an existing predictions file.

    python3 src/score.py results/LiveFlow_omni/window_response/predictions.jsonl

Scoring is a pure function of the recorded predictions, so a metric definition can be
changed and every past run re-summarised without re-querying any model.  Useful after a
crash, when merging shards, or to inspect one subtask of a large run.

``--sample-ids`` narrows a past run to a given set of rows, which is what makes two models'
columns comparable after the delivery sets have been regenerated:

    python3 src/score.py results/aura/sample30/predictions.jsonl \\
        --sample-ids results/moss_vl/matched174/predictions.jsonl \\
        --summary-out results/aura/matched174/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proactive_bench.dataset import write_json
from proactive_bench.metrics import format_table, summarize
from proactive_bench.runner import collect_records


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise LiveProBench predictions.")
    parser.add_argument("predictions", nargs="+", help="One or more predictions.jsonl files.")
    parser.add_argument("--summary-out", default="",
                        help="Where to write the summary. Defaults to summary.json beside "
                             "the first predictions file.")
    parser.add_argument("--subtask", action="append", default=[],
                        help="Keep only these subtasks. Repeatable.")
    parser.add_argument("--sample-ids", default="",
                        help="Keep only the rows whose sample_id appears in this file (one per "
                             "line, or a predictions/annotation jsonl to read them out of). "
                             "Use it to compare models over exactly the rows they all scored.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def read_sample_ids(path: str) -> set:
    """The sample_ids named by a file, whether it is a plain list or jsonl carrying them.

    Comparing two models means comparing them over the same rows.  That is not automatic here:
    the delivery sets are regenerated as subtasks are added, so ``--limit 30`` taken on two
    different days is two different samples, and a table built from those columns would read
    as a difference between models when part of it is a difference in rows.
    """
    ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                sample_id = json.loads(line).get("sample_id")
                if sample_id:
                    ids.add(str(sample_id))
            else:
                ids.add(line)
    return ids


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    records = []
    for path in args.predictions:
        if not os.path.exists(path):
            print(f"missing predictions file: {path}", file=sys.stderr)
            return 2
        records.extend(collect_records(path))
    if args.subtask:
        records = [item for item in records if str(item.get("subtask")) in set(args.subtask)]
    if args.sample_ids:
        if not os.path.exists(args.sample_ids):
            print(f"missing sample id file: {args.sample_ids}", file=sys.stderr)
            return 2
        keep = read_sample_ids(args.sample_ids)
        if not keep:
            print(f"{args.sample_ids} names no sample_id", file=sys.stderr)
            return 2
        before = len(records)
        records = [item for item in records if str(item.get("sample_id")) in keep]
        # Loud on purpose: a restriction that silently matched half of what was asked for
        # would produce a table that looks comparable and is not.
        if not args.quiet:
            print(f"rows       : {len(records)} of {before} kept, "
                  f"{len(keep)} id(s) requested")
    if not records:
        print("no records matched", file=sys.stderr)
        return 2

    summary = summarize(records)
    destination = args.summary_out or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions[0])), "summary.json"
    )
    write_json(destination, summary)
    if not args.quiet:
        print(format_table(summary))
        print(f"\nsummary: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
