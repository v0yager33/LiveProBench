#!/usr/bin/env python3
"""Cross-model comparison table.

    python3 src/report.py
    python3 src/report.py --run window_response --format csv --out results/window.csv

Walks ``results/<model>/<run>/summary.json`` and puts every model's headline rate in one
table, one row per subtask under its authoritative metric name.  Cells are left blank
rather than zero where a model has no rows for a subtask: a model that was never run on a
subtask and a model that failed it must not look alike.

Below the per-subtask rows come the subset scores and ``AVG``, which are the figures to quote.
The per-subtask rates cannot be averaged directly -- half of them reward speaking and half
reward staying quiet, so a model that only ever does one of the two scores well on its half and
mid-table on the mean.  Each subset pairs the two sides geometrically instead; see
``proactive_bench.metrics.subsets``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proactive_bench.suite import PROFILES, SUBSETS, SUBTASKS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LiveProBench runs across models.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model", action="append", default=[],
                        help="Keep only these model directories. Repeatable.")
    parser.add_argument("--run", action="append", default=[],
                        help="Keep only these run directories. Repeatable.")
    parser.add_argument("--metric", choices=("pass_rate", "pass_strict_rate"),
                        default="pass_rate",
                        help="pass_strict_rate additionally requires the content judge.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--out", default="", help="Write to this file instead of stdout.")
    return parser.parse_args(argv)


def discover(args: argparse.Namespace) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return (model, run, summary) for every summary.json under the results tree."""
    found: List[Tuple[str, str, Dict[str, Any]]] = []
    if not os.path.isdir(args.results_dir):
        return found
    for model in sorted(os.listdir(args.results_dir)):
        if args.model and model not in args.model:
            continue
        model_dir = os.path.join(args.results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for run in sorted(os.listdir(model_dir)):
            if args.run and run not in args.run:
                continue
            path = os.path.join(model_dir, run, "summary.json")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                found.append((model, run, json.load(handle)))
    return found


def headline(bucket: Dict[str, Any], metric: str) -> Optional[float]:
    """The rate a bucket reports, preferring the window verdict where it applies."""
    if metric == "pass_rate" and "in_window_rate" in bucket:
        return bucket["in_window_rate"]
    return bucket.get(metric)


def build_rows(entries, metric: str):
    """Return (columns, rows) where a row is subtask + one cell per column."""
    columns: List[str] = []
    cells: Dict[Tuple[str, str], Tuple[Optional[float], int]] = {}
    overall: Dict[str, Tuple[Optional[float], int, int]] = {}
    subset_cells: Dict[Tuple[str, str], Optional[float]] = {}
    subset_paired: Dict[str, bool] = {}
    averages: Dict[str, Optional[float]] = {}
    for model, run, summary in entries:
        column = model if len(entries) == len({item[0] for item in entries}) else f"{model}/{run}"
        if column not in columns:
            columns.append(column)
        for subtask, bucket in (summary.get("by_subtask") or {}).items():
            cells[(subtask, column)] = (headline(bucket, metric), int(bucket.get("n", 0)))
        for name, entry in (summary.get("by_subset") or {}).items():
            subset_cells[(name, column)] = entry.get("score")
            subset_paired[name] = bool(entry.get("paired"))
        averages[column] = (summary.get("headline") or {}).get("avg")
        # Not headline() here: the overall bucket also carries in_window_rate, aggregated over
        # the window rows alone, and letting it win would print a 14-row rate as the 50-row
        # headline.
        top = summary.get("overall") or {}
        overall[column] = (top.get(metric), int(top.get("n", 0)), int(top.get("errors", 0)))

    ordered = [name for name in SUBTASKS if any((name, column) in cells for column in columns)]
    ordered += sorted({subtask for subtask, _ in cells} - set(ordered))
    rows = []
    for subtask in ordered:
        item = SUBTASKS.get(subtask)
        row = {
            "profile": item.profile if item else "",
            "subtask": subtask,
            "metric": item.metric if item else "",
        }
        for column in columns:
            value, count = cells.get((subtask, column), (None, 0))
            row[column] = "" if value is None else f"{value:.4f}"
            row[f"{column} (n)"] = count or ""
        rows.append(row)

    # Appended in the order they should read, which is also the order the markdown view
    # recovers them in, so the two formats cannot drift apart.  Ordered by SUBSETS rather than
    # by profile: a subtask scored as its own subset has no entry in the profile list, and
    # would otherwise be appended alphabetically among the unknowns.
    subset_names = [name for name in SUBSETS if any(
        (name, column) in subset_cells for column in columns)]
    subset_names += sorted({name for name, _ in subset_cells} - set(subset_names))
    for name in subset_names:
        row = {
            "profile": "",
            "subtask": f"[{name}]",
            # No comma in either label: the csv quotes it correctly, but a reader reaching for
            # cut -d, on the output would silently lose a column.
            "metric": "sqrt(speak x silence)" if subset_paired.get(name)
                      else "speak side only (already joint)",
        }
        for column in columns:
            value = subset_cells.get((name, column))
            row[column] = "" if value is None else f"{value:.4f}"
            row[f"{column} (n)"] = ""
        rows.append(row)
    if subset_names:
        row = {"profile": "", "subtask": "AVG", "metric": "mean of the subset scores"}
        for column in columns:
            value = averages.get(column)
            row[column] = "" if value is None else f"{value:.4f}"
            row[f"{column} (n)"] = ""
        rows.append(row)

    # The headline every run prints, and the error count next to it.  Kept in the table rather
    # than left to the per-run summaries because a rate read without its errors is how a run
    # that dropped four rows to context overflow passed for a complete one.
    total = {"profile": "", "subtask": "OVERALL", "metric": metric}
    failed = {"profile": "", "subtask": "errors", "metric": "rows excluded from every rate"}
    for column in columns:
        value, count, errors = overall.get(column, (None, 0, 0))
        total[column] = "" if value is None else f"{value:.4f}"
        total[f"{column} (n)"] = count or ""
        failed[column] = str(errors)
        failed[f"{column} (n)"] = ""
    rows += [total, failed]
    return columns, rows


def as_markdown(columns: List[str], rows: List[Dict[str, Any]], metric: str) -> str:
    header = ["profile", "subtask", "metric", *columns]
    lines = [
        f"# LiveProBench -- {metric}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    profile_order = {name: index for index, name in enumerate(PROFILES)}
    # The trailing block (subsets, AVG, OVERALL, errors) carries no profile, so it sorts past
    # every subtask.  Its order is recovered from build_rows rather than restated here, so the
    # markdown and csv views cannot disagree about what comes after what.
    summary_order = {
        name: index for index, name in enumerate(
            row["subtask"] for row in rows if row["profile"] == ""
        )
    }
    for row in sorted(rows, key=lambda item: (profile_order.get(item["profile"], 99),
                                              summary_order.get(item["subtask"], -1),
                                              item["subtask"])):
        label = row["subtask"] if row["subtask"] in summary_order else f"`{row['subtask']}`"
        cells = [row["profile"], f"**{label}**" if row["profile"] == "" and label == row["subtask"]
                 else label, f"`{row['metric']}`" if row["metric"] else ""]
        for column in columns:
            count = row.get(f"{column} (n)")
            value = row.get(column, "")
            cells.append(f"{value} ({count})" if value and count else value or "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def as_csv(columns: List[str], rows: List[Dict[str, Any]], stream) -> None:
    fieldnames = ["profile", "subtask", "metric"]
    for column in columns:
        fieldnames += [column, f"{column} (n)"]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    entries = discover(args)
    if not entries:
        print(f"no summary.json found under {args.results_dir}", file=sys.stderr)
        return 2
    columns, rows = build_rows(entries, args.metric)

    if args.format == "markdown":
        text = as_markdown(columns, rows, args.metric)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return 0

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            as_csv(columns, rows, handle)
        print(f"wrote {args.out}")
    else:
        as_csv(columns, rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
