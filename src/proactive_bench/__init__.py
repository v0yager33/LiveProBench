"""LiveProBench -- does a streaming model speak up at the right moment?

One dataset row is one whole streaming conversation, never a flattened set of decision
points.  Three scoring modes cover the seven shipped subtasks -- four rows-in-a-window
subtasks, two two-sided ones, one counting -- and two further modes are defined and
unused, kept so that runs already published under them stay re-scorable;
``docs/protocol.md`` is the
authoritative description and :mod:`proactive_bench.suite` holds the subtask/metric
table that mirrors the generator's ``TASK_META``.
"""

SILENCE = "<|silence|>"

__version__ = "1.0.0"

__all__ = ["SILENCE", "__version__"]
