"""
benchmarks/timer.py
--------------------
Two timing utilities:

1. @timed decorator — wraps any function and prints elapsed time in ms.
2. benchmark()      — runs a function N times and returns mean elapsed ms.

Algorithmic concepts (Session 1 — functional programming patterns):
  - functools.wraps preserves the wrapped function's __name__ and __doc__
  - time.perf_counter() gives high-resolution wall-clock time (in seconds)
  - The mean of repeated runs reduces measurement noise (basic statistics)

Usage
-----
    from benchmarks import timed, benchmark

    @timed
    def my_function(x):
        return x ** 2

    my_function(42)
    # → my_function took 0.012 ms

    mean_ms = benchmark(my_function, 42, repeats=10)
    # → benchmark: my_function — mean 0.011 ms over 10 runs
"""

import time
import functools
from typing import Callable


def timed(fn: Callable) -> Callable:
    """
    Decorator that measures and prints the execution time of a function.

    Uses time.perf_counter() (high resolution, monotonic).
    Elapsed time is printed in milliseconds.
    functools.wraps preserves the original function's name and docstring.

    Example
    -------
        @timed
        def slow_sort(lst):
            return sorted(lst)

        slow_sort([3, 1, 2])
        # → slow_sort took 0.041 ms
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000  # seconds → ms
        print(f"{fn.__name__} took {elapsed_ms:.3f} ms")
        return result

    return wrapper


def benchmark(fn: Callable, *args, repeats: int = 5, **kwargs) -> float:
    """
    Run fn(*args, **kwargs) `repeats` times and return the mean elapsed
    time in milliseconds.

    Parameters
    ----------
    fn      : the function to benchmark.
    *args   : positional arguments passed to fn.
    repeats : number of repetitions (default 5). More repeats reduce noise.
    **kwargs: keyword arguments passed to fn.

    Returns
    -------
    float — mean elapsed time in milliseconds.

    Notes
    -----
    - The first run is sometimes slower due to Python caching / JIT effects.
      Using the mean over several runs gives a more representative figure.
    - time.perf_counter() returns seconds; we multiply by 1000 for ms.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    mean_ms = sum(times) / len(times)
    print(
        f"benchmark: {fn.__name__} — mean {mean_ms:.3f} ms "
        f"over {repeats} runs (min={min(times):.3f}, max={max(times):.3f})"
    )
    return mean_ms
