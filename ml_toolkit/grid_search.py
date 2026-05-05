"""
ml_toolkit/grid_search.py
--------------------------
Exhaustive hyperparameter grid search backed by HyperparamRegistry.

Uses itertools.product to enumerate all combinations (brute-force, Session 1).
Each trial result is stored in a BST via HyperparamRegistry.add_trial().

Example
-------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    def evaluate(params, dataset):
        X, y = dataset
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        return round(float(np.mean(scores)), 6)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }
    registry = grid_search(param_grid, evaluate, (X_train, y_train))
"""

import itertools
from typing import Callable

from tqdm import tqdm

from bst_toolkit import HyperparamRegistry


def grid_search(
    param_grid: dict,          # e.g. {"n_estimators": [50, 100], "max_depth": [3, 5]}
    evaluate_fn: Callable,     # function(params: dict, dataset) -> float
    dataset,                   # anything your evaluate_fn understands
    verbose: bool = True,      # show tqdm progress bar
) -> HyperparamRegistry:
    """
    Run an exhaustive grid search over all combinations in param_grid.

    Parameters
    ----------
    param_grid   : dict mapping parameter names to lists of candidate values.
    evaluate_fn  : callable(params: dict, dataset) -> float
                   Must return a single scalar score (higher = better).
    dataset      : passed as-is to evaluate_fn (e.g. a tuple (X, y)).
    verbose      : if True, display a tqdm progress bar.

    Returns
    -------
    HyperparamRegistry populated with one trial per combination.

    Algorithm (Session 1 — brute-force exhaustive search)
    -------------------------------------------------------
    itertools.product(*param_grid.values()) generates the Cartesian product
    of all value lists. Zipping with param_grid.keys() reconstructs each
    combination as a dict, which is passed directly to evaluate_fn.
    """
    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())

    # Total combinations = product of list lengths
    combinations = list(itertools.product(*value_lists))
    total = len(combinations)

    registry = HyperparamRegistry()

    iterator = tqdm(combinations, desc="Grid search", unit="trial") if verbose else combinations

    for values in iterator:
        params = dict(zip(keys, values))
        score = evaluate_fn(params, dataset)
        # Round to 6 decimal places to minimise floating-point collisions
        registry.add_trial(round(float(score), 6), params)

        if verbose:
            # Update tqdm postfix with current best
            best = registry.best()
            if best:
                iterator.set_postfix(best=f"{best.score:.4f}")  # type: ignore[union-attr]

    if verbose:
        best = registry.best()
        print(f"\nGrid search complete: {total} trials, best score = {best.score:.6f}")

    return registry
