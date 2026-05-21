"""HDC operation nodes — the data model behind the GUI counterparts.

Plan §16: every HDC primitive available in the REPL must also appear as a
draggable node in the GUI. This module defines the abstract NodeKind
catalog + a :class:`HdcNode` dataclass + a :class:`NodeGraph` execution
layer.

The GUI (Phase 4 plotly 3D + future Phase 12 native shell) consumes this
data model; the REPL also uses it for declarative computation construction:

    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 1})
    b = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 2})
    c = g.add(NodeKind.BIND, inputs=[a, b])
    result = g.execute()  # {node_id: Hypervector|float, ...}
    print(result[c])  # bound hypervector

Phase 17 (Maths Panel) consumes the same NodeGraph for advanced ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(Enum):
    """All HDC primitive kinds available as GUI nodes (plan §16)."""

    # Generators
    RANDOM = "random"

    # Core HDC operations (currently in plausiden-hdc)
    BIND = "bind"
    UNBIND = "unbind"
    BUNDLE = "bundle"
    PERMUTE = "permute"

    # Probes (HDC → scalar)
    COS_SIM = "cos_sim"
    HAMMING = "hamming"

    # Phase-17 extensions (stubs ready, native impls pending)
    INVERSE = "inverse"
    NEGATE = "negate"
    THRESHOLD = "threshold"
    HRR_BIND = "hrr_bind"
    HRR_UNBIND = "hrr_unbind"

    # Composite encodings (built atop core)
    POSITION = "position"
    ENCODE_SEQUENCE = "encode_sequence"
    ENCODE_SET = "encode_set"
    ENCODE_PAIR = "encode_pair"


@dataclass
class HdcNode:
    """One node in a HDC operation graph."""

    node_id: int
    kind: NodeKind
    inputs: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"HdcNode(id={self.node_id}, kind={self.kind.value}, inputs={self.inputs})"


class NodeGraphError(RuntimeError):
    """Raised when a node graph can't be executed."""


class NodeGraph:
    """A directed graph of HDC operation nodes with topological execution.

    Nodes are added with optional input references; execute() resolves the
    graph in topological order and returns a dict of node_id → result.

    The GUI (Phase 4 / Phase 12) renders these graphs as 2D / 3D node-link
    diagrams. The REPL uses them for declarative HDC composition.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, HdcNode] = {}
        self._next_id = 0

    def add(
        self,
        kind: NodeKind,
        *,
        inputs: list[int] | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Add a node; returns its assigned ID."""
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = HdcNode(
            node_id=node_id,
            kind=kind,
            inputs=list(inputs or []),
            params=dict(params or {}),
        )
        return node_id

    def get(self, node_id: int) -> HdcNode:
        if node_id not in self._nodes:
            raise NodeGraphError(f"no node with id {node_id}")
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[HdcNode]:
        """List all nodes in insertion order."""
        return list(self._nodes.values())

    def topological_order(self) -> list[int]:
        """Return node IDs in topological order (inputs always precede consumers).

        Raises NodeGraphError on cycles or dangling input references.
        """
        in_degree = dict.fromkeys(self._nodes, 0)
        for node in self._nodes.values():
            for inp in node.inputs:
                if inp not in self._nodes:
                    raise NodeGraphError(
                        f"node {node.node_id} references missing input {inp}"
                    )
                in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[int] = []
        # Simple Kahn's algorithm.
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for other in self._nodes.values():
                if nid in other.inputs:
                    in_degree[other.node_id] -= 1
                    if in_degree[other.node_id] == 0:
                        queue.append(other.node_id)
        if len(order) != len(self._nodes):
            raise NodeGraphError("cycle detected in node graph")
        return order

    def execute(self) -> dict[int, Any]:
        """Execute the graph in topological order; returns {node_id: result}.

        Requires the native graphnet extension. Probe nodes (cos_sim,
        hamming) return floats; everything else returns Hypervectors.
        """
        import graphnet

        if graphnet.Hypervector is None:
            raise NodeGraphError("native graphnet not installed")

        results: dict[int, Any] = {}
        for nid in self.topological_order():
            node = self._nodes[nid]
            results[nid] = _execute_node(node, results, graphnet)
        return results


def _execute_node(node: HdcNode, results: dict[int, Any], gn: Any) -> Any:
    """Dispatch one node to its underlying HDC operation."""
    kind = node.kind

    if kind is NodeKind.RANDOM:
        dim = int(node.params.get("dim", 10_000))
        seed = int(node.params.get("seed", 0))
        return gn.Hypervector.random(dim, seed)

    if kind is NodeKind.BIND:
        _expect_inputs(node, 2)
        return gn.bind(results[node.inputs[0]], results[node.inputs[1]])

    if kind is NodeKind.UNBIND:
        _expect_inputs(node, 2)
        return gn.unbind(results[node.inputs[0]], results[node.inputs[1]])

    if kind is NodeKind.BUNDLE:
        _expect_inputs_min(node, 1)
        return gn.bundle([results[i] for i in node.inputs])

    if kind is NodeKind.COS_SIM:
        _expect_inputs(node, 2)
        return gn.cos_sim(results[node.inputs[0]], results[node.inputs[1]])

    if kind is NodeKind.HAMMING:
        _expect_inputs(node, 2)
        return gn.hamming(results[node.inputs[0]], results[node.inputs[1]])

    if kind is NodeKind.NEGATE:
        # Bipolar negate: v ↔ -v. Native primitive not yet exposed.
        # Phase-17 follow-up: add `plausiden_hdc::negate` + wire here.
        _expect_inputs(node, 1)
        raise NodeGraphError(
            "negate node not yet implemented natively; deferred to Phase 17 "
            "extension"
        )

    if kind in (
        NodeKind.INVERSE,
        NodeKind.THRESHOLD,
        NodeKind.HRR_BIND,
        NodeKind.HRR_UNBIND,
        NodeKind.PERMUTE,
        NodeKind.POSITION,
        NodeKind.ENCODE_SEQUENCE,
        NodeKind.ENCODE_SET,
        NodeKind.ENCODE_PAIR,
    ):
        raise NodeGraphError(
            f"node kind `{kind.value}` not yet wired to a native op; "
            "deferred to Phase 17 extension tick"
        )

    raise NodeGraphError(f"unknown node kind: {kind}")


def _expect_inputs(node: HdcNode, n: int) -> None:
    if len(node.inputs) != n:
        raise NodeGraphError(
            f"node {node.node_id} ({node.kind.value}) expected {n} input(s), "
            f"got {len(node.inputs)}"
        )


def _expect_inputs_min(node: HdcNode, n: int) -> None:
    if len(node.inputs) < n:
        raise NodeGraphError(
            f"node {node.node_id} ({node.kind.value}) expected ≥ {n} input(s), "
            f"got {len(node.inputs)}"
        )


def available_kinds() -> list[tuple[str, bool]]:
    """List every NodeKind + whether it has a native impl wired today."""
    wired = {
        NodeKind.RANDOM,
        NodeKind.BIND,
        NodeKind.UNBIND,
        NodeKind.BUNDLE,
        NodeKind.COS_SIM,
        NodeKind.HAMMING,
    }
    return [(k.value, k in wired) for k in NodeKind]
