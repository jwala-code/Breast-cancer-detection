"""
bst_toolkit/node.py
--------------------
Defines TrialNode — a single node in the hyperparameter BST.
The BST is keyed by `score` (float).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrialNode:
    """
    A single node in the hyperparameter BST.
    The BST is keyed by `score`.

    Attributes
    ----------
    score  : the evaluation metric for this trial (e.g. accuracy)
    params : the hyperparameter dictionary used in this trial
    left   : left child node (score strictly less than this node)
    right  : right child node (score strictly greater than this node)
    """

    score: float
    params: dict
    left: Optional["TrialNode"] = field(default=None, repr=False)
    right: Optional["TrialNode"] = field(default=None, repr=False)

    def __lt__(self, other: "TrialNode") -> bool:
        """Return True if this node's score is less than the other's."""
        return self.score < other.score

    def __repr__(self) -> str:
        """Return a readable string representation of the node."""
        return f"TrialNode(score={self.score:.4f}, params={self.params})"
