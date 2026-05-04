"""
bst_toolkit/registry.py
------------------------
HyperparamRegistry: high-level interface around BST for managing
hyperparameter search trials.

Provides:
  - add_trial()     — insert a scored trial
  - best() / worst() — find extremes
  - top_k()         — k highest-scoring configs
  - range_query()   — all configs within [lo, hi]
  - prune_below()   — remove weak trials
  - summary()       — statistics overview
"""

from __future__ import annotations

from typing import List, Optional

from .bst import BST
from .node import TrialNode


class HyperparamRegistry:
    """
    High-level interface around BST for managing hyperparameter trials.
    Provides range queries, top-k retrieval, pruning, and summaries.
    """

    def __init__(self) -> None:
        self._bst: BST = BST()
        # History log keeps insertion order (including duplicates for analysis)
        self._history: List[dict] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def add_trial(self, score: float, params: dict) -> None:
        """
        Record a new trial: insert into BST and append to history log.

        Collision handling: if a trial with this exact score already exists
        in the BST, the existing params are kept (first-inserted wins) and
        this call is ignored silently. Round scores to 6 decimals before
        calling this method to minimise accidental collisions from
        floating-point noise.
        """
        self._history.append({"score": score, "params": params})
        self._bst.insert(score, params)

    def best(self) -> Optional[TrialNode]:
        """Return the highest-scoring trial. Uses BST find_max()."""
        return self._bst.find_max()

    def worst(self) -> Optional[TrialNode]:
        """Return the lowest-scoring trial. Uses BST find_min()."""
        return self._bst.find_min()

    def top_k(self, k: int) -> List[TrialNode]:
        """
        Return the k highest-scoring trials in descending order.
        Uses a reverse in-order traversal (Right → Node → Left).
        Complexity: O(k + h).
        """
        result: List[TrialNode] = []
        self._reverse_inorder(self._bst.root, result, k)
        return result

    def range_query(self, lo: float, hi: float) -> List[TrialNode]:
        """
        Return all trials with lo <= score <= hi, sorted ascending.
        Uses BST pruning — doesn't explore a subtree if it can't
        contain values in the range.
        Complexity: O(k + h) where k = number of results.
        """
        result: List[TrialNode] = []
        self._range(self._bst.root, lo, hi, result)
        return result

    def prune_below(self, threshold: float) -> int:
        """
        Delete all trials with score < threshold.
        Returns the count of deleted nodes.
        Collects scores to delete via inorder(), then deletes one by one.
        """
        to_delete = [
            node.score for node in self._bst.inorder()
            if node.score < threshold
        ]
        for score in to_delete:
            self._bst.delete(score)
        return len(to_delete)

    def all_trials(self) -> List[TrialNode]:
        """Return all trials sorted ascending by score (in-order)."""
        return self._bst.inorder()

    def summary(self) -> dict:
        """
        Return a dict with: count, best_score, worst_score,
        mean_score, tree_height, is_balanced.
        """
        nodes = self.all_trials()
        if not nodes:
            return {
                "count": 0,
                "best_score": None,
                "worst_score": None,
                "mean_score": None,
                "tree_height": 0,
                "is_balanced": True,
            }
        scores = [n.score for n in nodes]
        return {
            "count": len(nodes),
            "best_score": round(max(scores), 6),
            "worst_score": round(min(scores), 6),
            "mean_score": round(sum(scores) / len(scores), 6),
            "tree_height": self._bst.height(),
            "is_balanced": self._bst.is_balanced(),
        }

    # Convenience pass-throughs to the underlying BST
    def is_balanced(self) -> bool:
        return self._bst.is_balanced()

    def height(self) -> int:
        return self._bst.height()

    def __len__(self) -> int:
        return len(self._bst)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reverse_inorder(
        self,
        node: Optional[TrialNode],
        result: List[TrialNode],
        k: int,
    ) -> None:
        """
        Right → Node → Left traversal. Stops when len(result) == k.
        Visits nodes in descending score order, so the first k nodes
        are the k highest-scoring trials.
        """
        if node is None or len(result) == k:
            return
        # Go right first (higher scores)
        self._reverse_inorder(node.right, result, k)
        if len(result) < k:
            result.append(node)
            # Then go left (lower scores)
            self._reverse_inorder(node.left, result, k)

    def _range(
        self,
        node: Optional[TrialNode],
        lo: float,
        hi: float,
        result: List[TrialNode],
    ) -> None:
        """
        Collect all nodes with lo <= score <= hi.
        Prune: only go left if node.score > lo;
               only go right if node.score < hi.
        """
        if node is None:
            return

        # Prune left subtree: no point descending if current score <= lo
        if node.score > lo:
            self._range(node.left, lo, hi, result)

        # Visit current node if in range
        if lo <= node.score <= hi:
            result.append(node)

        # Prune right subtree: no point descending if current score >= hi
        if node.score < hi:
            self._range(node.right, lo, hi, result)
