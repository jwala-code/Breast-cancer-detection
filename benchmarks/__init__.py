"""
benchmarks
----------
Timing utilities for measuring execution performance.

Public API:
    from benchmarks import timed, benchmark
"""

from .timer import timed, benchmark

__all__ = ["timed", "benchmark"]
