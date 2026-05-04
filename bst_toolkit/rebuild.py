"""
bst_toolkit/rebuild.py
-----------------------
Three strategies for rebuilding a HyperparamRegistry on a new dataset.

Strategy 1 — rebuild_naive:
    Insert re-scored trials in sorted order → degenerate O(n²) tree.
    Intentionally bad; used to demonstrate the worst case.

Strategy 2 — rebuild_shuffled:
    Shuffle before inserting → expected O(n log n), not guaranteed balanced.

Strategy 3 — rebuild_balanced:
    Sort by new score then build perfectly balanced BST via divide & conquer.
    Guaranteed height = floor(log₂ n). Recommended for Phase 2.
"""

from __future__ import annotations

import random
from typing import List, Callable, Tuple, Optional

from .registry import HyperparamRegistry
from .node import TrialNode


# ── Strategy 1: Naive (sorted insertion → degenerate tree) ────────────────────

def rebuild_naive(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset,
) -> HyperparamRegistry:
    """
    Strategy 1 — Re-score every trial and insert one by one.

    WARNING: all_trials() returns nodes in sorted (ascending) order.
    Inserting a sorted sequence into a BST produces a degenerate tree
    (like a linked list), giving O(n²) total time instead of O(n log n).
    This is intentional — you will measure and demonstrate this problem.
    """
    new_registry = HyperparamRegistry()

    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(round(new_score, 6), node.params)

    return new_registry


# ── Strategy 2: Shuffled (random insertion → expected balanced) ───────────────

def rebuild_shuffled(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset,
) -> HyperparamRegistry:
    """
    Strategy 2 — Shuffle trials before re-inserting.

    Breaking the sorted order prevents degenerate insertion.
    Expected O(n log n) but the resulting tree is not guaranteed balanced.
    """
    trials = registry.all_trials()
    random.shuffle(trials)  # in-place shuffle breaks sorted order

    new_registry = HyperparamRegistry()
    for node in trials:
        new_score = evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(round(new_score, 6), node.params)

    return new_registry


# ── Strategy 3: Balanced (divide & conquer → perfect balance) ─────────────────

def rebuild_balanced(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset,
) -> HyperparamRegistry:
    """
    Strategy 3 — Build a perfectly balanced BST using divide & conquer.

    Steps:
    1. Re-score all trials.
    2. Sort by new score — O(n log n).
    3. Call _build_from_sorted() — O(n), guarantees height = floor(log2 n).

    This is the recommended strategy for Phase 2.
    """
    # Step 1: re-score all trials
    scored_pairs: List[Tuple[float, dict]] = []
    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)
        scored_pairs.append((round(new_score, 6), node.params))

    # Step 2: sort by new score ascending
    scored_pairs.sort(key=lambda x: x[0])

    # Step 3: build a balanced BST from the sorted list
    new_registry = HyperparamRegistry()
    root = _build_from_sorted(scored_pairs)
    # Walk the balanced tree and populate the new registry via add_trial
    # (preserving the balanced structure by inserting in level order)
    _insert_level_order(root, new_registry)

    return new_registry


def _build_from_sorted(
    sorted_trials: List[Tuple[float, dict]],
) -> Optional[TrialNode]:
    """
    Recursively build a balanced BST from a sorted list of (score, params).

    Algorithm (Divide & Conquer — same structure as merge sort):
    - Base case: empty list → return None
    - Find the middle element → make it the root
    - Recurse on the left half  → left subtree
    - Recurse on the right half → right subtree

    Complexity: O(n) time, O(log n) stack space.
    """
    if not sorted_trials:
        return None

    mid = len(sorted_trials) // 2
    score, params = sorted_trials[mid]
    node = TrialNode(score=score, params=params)

    node.left = _build_from_sorted(sorted_trials[:mid])
    node.right = _build_from_sorted(sorted_trials[mid + 1:])

    return node


def _insert_level_order(root: Optional[TrialNode], registry: HyperparamRegistry) -> None:
    """
    Walk a pre-built tree in level order and add each node to the registry.
    This preserves the balanced structure because we insert the root first,
    then children — mimicking how a balanced BST would be populated.
    """
    if root is None:
        return

    import collections
    queue = collections.deque([root])

    while queue:
        node = queue.popleft()
        registry.add_trial(node.score, node.params)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
