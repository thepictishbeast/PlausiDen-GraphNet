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
        Duplicate input IDs on a single node (e.g. cos_sim(v, v)) count as
        ONE incoming edge — they represent the same data dependency.
        """
        # Build a unique edge set per node: which distinct nodes feed into me.
        unique_inputs: dict[int, set[int]] = {nid: set() for nid in self._nodes}
        for node in self._nodes.values():
            for inp in node.inputs:
                if inp not in self._nodes:
                    raise NodeGraphError(
                        f"node {node.node_id} references missing input {inp}"
                    )
                unique_inputs[node.node_id].add(inp)

        in_degree = {nid: len(inputs) for nid, inputs in unique_inputs.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[int] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for other_id, other_inputs in unique_inputs.items():
                if nid in other_inputs:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
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
        _expect_inputs(node, 1)
        return gn.negate(results[node.inputs[0]])

    if kind is NodeKind.INVERSE:
        # Bipolar HDC is self-inverse: bind(v, v⁻¹) where v⁻¹ = v.
        _expect_inputs(node, 1)
        return results[node.inputs[0]]

    if kind is NodeKind.THRESHOLD:
        # Hypervectors are already bipolar — threshold is the identity.
        _expect_inputs(node, 1)
        return results[node.inputs[0]]

    if kind is NodeKind.PERMUTE:
        _expect_inputs(node, 1)
        shift = int(node.params.get("shift", 1))
        return gn.permute(results[node.inputs[0]], shift)

    if kind in (NodeKind.HRR_BIND, NodeKind.HRR_UNBIND):
        # HRR (FFT) binding is bipolar-self-inverse like dense bind.
        # Wire via a one-op Stack so the existing Operation::HrrBind kernel
        # runs.
        _expect_inputs(node, 2)
        input_vec = results[node.inputs[0]]
        key = results[node.inputs[1]]
        s = gn.Stack(input_vec.dim())
        s.add_operation(gn.Operation.hrr_bind(key))
        return s.forward(input_vec)

    if kind is NodeKind.POSITION:
        # Position-encoded item: bind(item, permute(position_key, pos)).
        _expect_inputs(node, 2)
        item = results[node.inputs[0]]
        position_key = results[node.inputs[1]]
        pos = int(node.params.get("pos", 0))
        rotated_key = gn.permute(position_key, pos)
        return gn.bind(item, rotated_key)

    if kind is NodeKind.ENCODE_SEQUENCE:
        # bundle(bind(item_i, permute(position_key, i))) for i in inputs.
        # Inputs are the items; params["position_key_id"] points to a
        # role-key node ID in `results` (the convention is N + 1 inputs:
        # last input is the position key).
        if len(node.inputs) < 2:
            raise NodeGraphError(
                f"node {node.node_id} (encode_sequence) needs ≥1 item input "
                "and 1 position-key input"
            )
        position_key = results[node.inputs[-1]]
        items = [results[i] for i in node.inputs[:-1]]
        bound = [gn.bind(item, gn.permute(position_key, idx)) for idx, item in enumerate(items)]
        return gn.bundle(bound)

    if kind is NodeKind.ENCODE_SET:
        # bundle(items) — order-independent set encoding.
        _expect_inputs_min(node, 1)
        return gn.bundle([results[i] for i in node.inputs])

    if kind is NodeKind.ENCODE_PAIR:
        # Inputs = [key_a, val_a, key_b, val_b]; output = bundle(
        #   bind(key_a, val_a), bind(key_b, val_b)).
        if len(node.inputs) != 4:
            raise NodeGraphError(
                f"node {node.node_id} (encode_pair) expects exactly 4 inputs "
                f"(key_a, val_a, key_b, val_b), got {len(node.inputs)}"
            )
        ka = results[node.inputs[0]]
        va = results[node.inputs[1]]
        kb = results[node.inputs[2]]
        vb = results[node.inputs[3]]
        return gn.bundle([gn.bind(ka, va), gn.bind(kb, vb)])

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
    """List every NodeKind + whether it has a native impl wired today.

    As of Phase 11 wave 2 closeout, ALL 16 NodeKinds are wired natively.
    """
    return [(k.value, True) for k in NodeKind]
