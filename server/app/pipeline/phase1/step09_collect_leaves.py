"""
PIPELINE.md step 9 — collect leaves and ancestor frame chains.

After step 8's recursion returns the fully-populated `SubsceneNode`
tree, the orchestrator needs two flat views of it for phase 2:

  * `collect_leaves(root)` — every atomic leaf, in depth-first order.
  * `collect_frames_for_leaf(root, leaf)` — the leaf's own frames plus
    every ancestor's frames on the root→leaf chain. This is the frame
    list the leaf inherits and that phase 2 passes into step 10 + 11
    validators.
"""

from __future__ import annotations

from app.core.types import Frame, SubsceneNode


def collect_leaves(root: SubsceneNode) -> list[SubsceneNode]:
    """Flatten the tree into a depth-first list of atomic leaves."""
    out: list[SubsceneNode] = []
    _walk_leaves(root, out)
    return out


def _walk_leaves(node: SubsceneNode, out: list[SubsceneNode]) -> None:
    if node.is_atomic:
        out.append(node)
        return
    for child in node.children:
        _walk_leaves(child, out)


def collect_frames_for_leaf(root: SubsceneNode, leaf: SubsceneNode) -> list[Frame]:
    """Return the leaf's frames plus every ancestor's frames up to root.
    Returns `[]` if no frames on the path.
    """
    chain = _find_ancestors(root, leaf.scope_id)
    frames: list[Frame] = []
    for node in chain:
        frames.extend(node.frames)
    return frames


def _find_ancestors(node: SubsceneNode, target_scope_id: str) -> list[SubsceneNode]:
    """Return the chain [root, ..., target_node] or [] if not found."""
    if node.scope_id == target_scope_id:
        return [node]
    for child in node.children:
        inner = _find_ancestors(child, target_scope_id)
        if inner:
            return [node, *inner]
    return []
