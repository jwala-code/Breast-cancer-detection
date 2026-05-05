"""
ml_toolkit/transfer.py
-----------------------
Transfer analysis: compares the rank ordering of hyperparameter configurations
between two datasets (registry_A and registry_B).

Algorithmic concepts (from course sessions):
  - In-order BST traversal → sorted ranking in O(n)  [Session 5]
  - Hash table (dict) for O(1) rank lookups           [Session 1]
  - sorted() with a key for rank-drift ordering       [Session 3]

Output: a list of dicts with keys:
  params, score_A, score_B, rank_A, rank_B,
  drift  (= rank_A − rank_B, positive means improved in B),
  transfer  ("✓ good" or "✗ poor")
"""

from __future__ import annotations

from typing import List

from bst_toolkit import HyperparamRegistry


# A config is considered a "good transfer" if its absolute rank drift
# is within this fraction of the total number of configs.
_TRANSFER_THRESHOLD_FRACTION = 0.25


def analyse_transfer(
    registry_A: HyperparamRegistry,
    registry_B: HyperparamRegistry,
    transfer_threshold: float | None = None,
) -> List[dict]:
    """
    Compare the ranking of every configuration between two registries.

    Both registries must contain results for the *same* set of
    hyperparameter combinations (same params dicts), evaluated on
    different datasets.

    Parameters
    ----------
    registry_A          : registry built on Dataset A.
    registry_B          : registry built on Dataset B.
    transfer_threshold  : max |drift| (as fraction of n) to label a
                          config as "✓ good". Defaults to 0.25 × n.

    Returns
    -------
    List of dicts sorted by drift descending (best improvers first).
    Each dict has: params, score_A, score_B, rank_A, rank_B, drift, transfer.

    Algorithm
    ---------
    1. In-order traversal of registry_A → nodes sorted ascending.
       Assign rank_A[i] = i+1 (rank 1 = worst, rank n = best).
       Build a lookup dict keyed by a hashable params signature.

    2. Same for registry_B → rank_B lookup.

    3. For each config present in both, compute drift = rank_B − rank_A.
       (Positive drift means the config moved *up* in B — it improved.)

    4. Label as "✓ good" if |drift| <= threshold, else "✗ poor".

    5. Sort the final report by drift descending.
    """
    nodes_A = registry_A.all_trials()  # ascending score order
    nodes_B = registry_B.all_trials()

    n = max(len(nodes_A), len(nodes_B), 1)
    threshold = transfer_threshold if transfer_threshold is not None else _TRANSFER_THRESHOLD_FRACTION * n

    # ── Step 1: build rank lookup for A ──────────────────────────────────────
    # rank 1 = lowest score (worst), rank n = highest score (best)
    rank_A: dict = {}
    score_A_lookup: dict = {}
    for rank, node in enumerate(nodes_A, start=1):
        key = _params_key(node.params)
        rank_A[key] = rank
        score_A_lookup[key] = node.score

    # ── Step 2: build rank lookup for B ──────────────────────────────────────
    rank_B: dict = {}
    score_B_lookup: dict = {}
    for rank, node in enumerate(nodes_B, start=1):
        key = _params_key(node.params)
        rank_B[key] = rank
        score_B_lookup[key] = node.score

    # ── Step 3 & 4: compute drift and label ──────────────────────────────────
    report: List[dict] = []

    # Use the union of both key sets to catch configs missing from either side
    all_keys = set(rank_A.keys()) | set(rank_B.keys())

    for key in all_keys:
        if key not in rank_A or key not in rank_B:
            # Config only appears in one registry — skip (no meaningful transfer)
            continue

        r_A = rank_A[key]
        r_B = rank_B[key]
        drift = r_B - r_A  # positive = improved in B

        transfer_label = "✓ good" if abs(drift) <= threshold else "✗ poor"

        report.append({
            "params": _key_to_params(key),
            "score_A": score_A_lookup[key],
            "score_B": score_B_lookup[key],
            "rank_A": r_A,
            "rank_B": r_B,
            "drift": drift,
            "transfer": transfer_label,
        })

    # ── Step 5: sort by drift descending (best improvers first) ──────────────
    report = sorted(report, key=lambda x: x["drift"], reverse=True)

    return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _params_key(params: dict) -> tuple:
    """
    Convert a params dict to a hashable, order-independent key.
    Uses a sorted tuple of (key, value) pairs so that dict ordering
    doesn't affect matching.
    """
    return tuple(sorted((k, str(v)) for k, v in params.items()))


def _key_to_params(key: tuple) -> dict:
    """Reconstruct a params dict from its tuple key (best-effort)."""
    result = {}
    for k, v in key:
        # Try to recover original types (int, float, None) from string
        if v == "None":
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result
