"""
bst_toolkit/bst.py
-------------------
Binary Search Tree (BST) keyed by trial score.

BST property: left child < parent < right child.
All single-path operations are O(h) where h = tree height.
Traversals are O(n).
"""

from __future__ import annotations

import collections
from typing import Optional, List

from .node import TrialNode


class BST:
    """
    Binary Search Tree keyed by trial score.

    Left child < parent < right child (BST property).
    All operations are O(h) where h = tree height.
    """

    def __init__(self) -> None:
        self.root: Optional[TrialNode] = None
        self._size: int = 0

    # ── Public methods ────────────────────────────────────────────────────────

    def insert(self, score: float, params: dict) -> None:
        """
        Insert a new trial into the BST.

        If a node with the same score already exists, KEEP THE EXISTING
        PARAMS (first-inserted wins). The duplicate is silently ignored.

        Complexity: O(h) average.

        Note: to minimise collisions in practice, round scores to 6 decimals
        before calling insert, e.g. registry.add_trial(round(score, 6), params).
        """
        old_size = self._size
        self.root = self._insert(self.root, score, params)
        # _insert returns without creating a new node on duplicate,
        # so we check whether the tree actually grew.
        if self._size == old_size:
            # Node was a duplicate — increment was not done inside _insert,
            # so we count it here only for NEW nodes.
            pass  # size already managed inside _insert

    def delete(self, score: float) -> None:
        """
        Delete the node with the given score.
        Apply the correct case (0, 1, or 2 children).
        Complexity: O(h).
        """
        self.root, was_deleted = self._delete(self.root, score)
        if was_deleted:
            self._size -= 1

    def search(self, score: float) -> Optional[TrialNode]:
        """
        Return the node with this score, or None if not found.
        Complexity: O(h).
        """
        return self._search(self.root, score)

    def find_min(self) -> Optional[TrialNode]:
        """
        Return the node with the lowest score (leftmost node).
        Complexity: O(h).
        """
        if self.root is None:
            return None
        return self._find_min(self.root)

    def find_max(self) -> Optional[TrialNode]:
        """
        Return the node with the highest score (rightmost node).
        Complexity: O(h).
        """
        if self.root is None:
            return None
        node = self.root
        while node.right is not None:
            node = node.right
        return node

    def height(self) -> int:
        """
        Return the height of the tree (0 for an empty tree).
        Complexity: O(n).
        """
        return self._height(self.root)

    def is_balanced(self) -> bool:
        """
        Return True if the tree is balanced:
        |height(left) - height(right)| <= 1 at every node.
        """
        return self._check_balanced(self.root) != -1

    def __len__(self) -> int:
        return self._size

    # ── Traversals ────────────────────────────────────────────────────────────

    def inorder(self) -> List[TrialNode]:
        """
        In-order traversal: Left → Node → Right.
        Returns nodes sorted by score ASCENDING.
        This is the key property of BST in-order traversal.
        Complexity: O(n).
        """
        result: List[TrialNode] = []
        self._inorder(self.root, result)
        return result

    def preorder(self) -> List[TrialNode]:
        """
        Pre-order traversal: Node → Left → Right.
        Returns the root first — used to serialise/copy the tree.
        Complexity: O(n).
        """
        result: List[TrialNode] = []
        self._preorder(self.root, result)
        return result

    def postorder(self) -> List[TrialNode]:
        """
        Post-order traversal: Left → Right → Node.
        Returns the root last — used to delete the tree safely.
        Complexity: O(n).
        """
        result: List[TrialNode] = []
        self._postorder(self.root, result)
        return result

    def level_order(self) -> List[List[TrialNode]]:
        """
        Breadth-first traversal: level by level, left to right.
        Uses a queue (collections.deque), NOT recursion.
        Returns a list of levels: [[root], [level1_nodes], ...].
        Complexity: O(n).
        """
        if self.root is None:
            return []

        levels: List[List[TrialNode]] = []
        queue = collections.deque([self.root])

        while queue:
            level_size = len(queue)
            current_level: List[TrialNode] = []

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)

            levels.append(current_level)

        return levels

    # ── Private helpers ───────────────────────────────────────────────────────

    def _insert(self, node: Optional[TrialNode], score: float, params: dict) -> TrialNode:
        """
        Recursive insert helper.
        - If node is None: create and return a new TrialNode.
        - If score < node.score: recurse left.
        - If score > node.score: recurse right.
        - If score == node.score: DO NOTHING — keep the existing params
          (first-inserted wins). The duplicate is silently ignored.
        Always return node.
        """
        if node is None:
            # New node — increment size here
            self._size += 1
            return TrialNode(score=score, params=params)

        if score < node.score:
            node.left = self._insert(node.left, score, params)
        elif score > node.score:
            node.right = self._insert(node.right, score, params)
        # else: duplicate score — do nothing (first-inserted wins)

        return node

    def _delete(self, node: Optional[TrialNode], score: float):
        """
        Recursive delete helper. Returns (node, was_deleted: bool).

        Three cases:
        1. Leaf (no children): return None, True
        2. One child: return the existing child, True
        3. Two children: find in-order successor (min of right subtree),
           copy its value into this node, delete it from the right subtree.
        """
        if node is None:
            return None, False

        if score < node.score:
            node.left, deleted = self._delete(node.left, score)
            return node, deleted
        elif score > node.score:
            node.right, deleted = self._delete(node.right, score)
            return node, deleted
        else:
            # Found the node to delete
            # Case 1 & 2: zero or one child
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True

            # Case 3: two children
            # Find in-order successor (minimum of right subtree)
            successor = self._find_min(node.right)
            # Copy successor's data into current node
            node.score = successor.score
            node.params = successor.params
            # Delete the successor from the right subtree
            node.right, _ = self._delete(node.right, successor.score)
            return node, True

    def _search(self, node: Optional[TrialNode], score: float) -> Optional[TrialNode]:
        """Recursive search: returns the node with matching score or None."""
        if node is None:
            return None
        if score == node.score:
            return node
        if score < node.score:
            return self._search(node.left, score)
        return self._search(node.right, score)

    def _find_min(self, node: TrialNode) -> TrialNode:
        """Traverse left until node.left is None — returns the minimum node."""
        while node.left is not None:
            node = node.left
        return node

    def _height(self, node: Optional[TrialNode]) -> int:
        """Recursive height: 0 for None, else 1 + max(left, right)."""
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _check_balanced(self, node: Optional[TrialNode]) -> int:
        """
        Returns the height of the subtree if balanced, or -1 if not.
        A node is unbalanced if |left_height - right_height| > 1.
        Uses -1 as a sentinel for 'unbalanced' (avoids a second pass).
        """
        if node is None:
            return 0

        left_h = self._check_balanced(node.left)
        if left_h == -1:
            return -1  # already found imbalance below

        right_h = self._check_balanced(node.right)
        if right_h == -1:
            return -1

        if abs(left_h - right_h) > 1:
            return -1  # this node is unbalanced

        return 1 + max(left_h, right_h)

    def _inorder(self, node: Optional[TrialNode], result: List[TrialNode]) -> None:
        """Left → Node → Right. Appends nodes to result in ascending score order."""
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node)
        self._inorder(node.right, result)

    def _preorder(self, node: Optional[TrialNode], result: List[TrialNode]) -> None:
        """Node → Left → Right. Root-first ordering."""
        if node is None:
            return
        result.append(node)
        self._preorder(node.left, result)
        self._preorder(node.right, result)

    def _postorder(self, node: Optional[TrialNode], result: List[TrialNode]) -> None:
        """Left → Right → Node. Root-last ordering."""
        if node is None:
            return
        self._postorder(node.left, result)
        self._postorder(node.right, result)
        result.append(node)
