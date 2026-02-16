"""Instruction dependency DAG.

Builds a directed acyclic graph from PTX instructions using
register def-use analysis. Supports RAW, WAR, and WAW edges,
critical path computation, and ready-set tracking for scheduling.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .instruction import Instruction


class DepType(Enum):
    """Dependency edge types."""
    RAW = 'RAW'   # Read After Write: consumer reads what producer wrote
    WAR = 'WAR'   # Write After Read: writer overwrites what reader used
    WAW = 'WAW'   # Write After Write: second write overwrites first write


@dataclass
class Edge:
    """A dependency edge in the instruction DAG."""
    src: int      # Producer instruction ID
    dst: int      # Consumer instruction ID
    dep_type: DepType
    register: str  # The register causing the dependency

    def __repr__(self):
        return f"Edge({self.src}->{self.dst} {self.dep_type.value} via {self.register})"


class InstructionDAG:
    """Dependency DAG over PTX instructions.

    Nodes are instruction IDs (0..N-1). Edges encode data dependencies
    derived from register def-use chains.

    Key properties computed at construction:
      - predecessors/successors adjacency lists
      - depth (longest path from any root to this node)
      - height (longest path from this node to any leaf)
      - critical_path_length (depth of deepest leaf)
      - roots (nodes with no predecessors)
      - leaves (nodes with no successors)
    """

    def __init__(self, instructions: List[Instruction]):
        self.instructions = {inst.id: inst for inst in instructions}
        self.n = len(instructions)
        self.edges: List[Edge] = []

        # Adjacency lists: instruction_id -> set of instruction_ids
        self.successors: Dict[int, Set[int]] = defaultdict(set)
        self.predecessors: Dict[int, Set[int]] = defaultdict(set)

        # Edge details: (src, dst) -> list of edges (may have multiple deps)
        self.edge_map: Dict[Tuple[int, int], List[Edge]] = defaultdict(list)

        # Build dependencies from register analysis
        self._build_dependencies(instructions)

        # Compute graph properties
        self.roots: List[int] = [i for i in range(self.n) if not self.predecessors[i]]
        self.leaves: List[int] = [i for i in range(self.n) if not self.successors[i]]

        # Longest-path from roots (depth) and to leaves (height)
        self.depth: Dict[int, int] = {}
        self.height: Dict[int, int] = {}
        self._compute_depth()
        self._compute_height()

        self.critical_path_length = max(self.depth.values()) if self.depth else 0

    def _build_dependencies(self, instructions: List[Instruction]):
        """Build RAW, WAR, and WAW edges from register def-use chains.

        For each register, track the last writer and last readers.
        When a new instruction uses that register, create edges:
          - Reads register: RAW edge from last writer (if any)
          - Writes register: WAR edges from all last readers, WAW from last writer
        """
        # Track last writer (instruction ID) per register
        last_writer: Dict[str, int] = {}
        # Track readers since last write per register
        last_readers: Dict[str, Set[int]] = defaultdict(set)

        for inst in instructions:
            inst_id = inst.id

            # Process source registers (reads)
            for reg in inst.src_regs:
                # RAW: this instruction reads a register written earlier
                if reg in last_writer:
                    writer_id = last_writer[reg]
                    if writer_id != inst_id:
                        self._add_edge(writer_id, inst_id, DepType.RAW, reg)
                last_readers[reg].add(inst_id)

            # Process destination registers (writes)
            for reg in inst.dest_regs:
                # WAR: this instruction writes a register read earlier
                for reader_id in last_readers.get(reg, set()):
                    if reader_id != inst_id:
                        self._add_edge(reader_id, inst_id, DepType.WAR, reg)

                # WAW: this instruction writes a register written earlier
                if reg in last_writer:
                    writer_id = last_writer[reg]
                    if writer_id != inst_id:
                        self._add_edge(writer_id, inst_id, DepType.WAW, reg)

                # Update tracking: this instruction is now the last writer
                last_writer[reg] = inst_id
                last_readers[reg] = set()  # Clear readers since we have a new writer

    def _add_edge(self, src: int, dst: int, dep_type: DepType, register: str):
        """Add a dependency edge, avoiding duplicates for the same (src, dst, type)."""
        # Check for duplicate
        for existing in self.edge_map[(src, dst)]:
            if existing.dep_type == dep_type and existing.register == register:
                return

        edge = Edge(src=src, dst=dst, dep_type=dep_type, register=register)
        self.edges.append(edge)
        self.edge_map[(src, dst)].append(edge)
        self.successors[src].add(dst)
        self.predecessors[dst].add(src)

    def _compute_depth(self):
        """Compute longest path from any root to each node (weighted by latency)."""
        # Topological order via Kahn's algorithm
        order = self.topological_sort()

        for node_id in range(self.n):
            self.depth[node_id] = 0

        for node_id in order:
            inst = self.instructions[node_id]
            node_cost = inst.latency
            for succ_id in self.successors[node_id]:
                new_depth = self.depth[node_id] + node_cost
                if new_depth > self.depth[succ_id]:
                    self.depth[succ_id] = new_depth

    def _compute_height(self):
        """Compute longest path from each node to any leaf (weighted by latency)."""
        order = self.topological_sort()

        for node_id in range(self.n):
            self.height[node_id] = self.instructions[node_id].latency

        # Reverse topological order
        for node_id in reversed(order):
            inst = self.instructions[node_id]
            for succ_id in self.successors[node_id]:
                new_height = self.height[succ_id] + inst.latency
                if new_height > self.height[node_id]:
                    self.height[node_id] = new_height

    def topological_sort(self) -> List[int]:
        """Kahn's algorithm. Returns instruction IDs in topological order."""
        in_degree = {i: len(self.predecessors[i]) for i in range(self.n)}
        queue = [i for i in range(self.n) if in_degree[i] == 0]
        order = []

        while queue:
            # Stable sort: pick lowest ID among ready nodes
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for succ in self.successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != self.n:
            raise ValueError(
                f"Cycle detected in DAG: sorted {len(order)} of {self.n} nodes"
            )
        return order

    def get_ready_set(self, scheduled: FrozenSet[int]) -> List[int]:
        """Get instructions whose predecessors are all scheduled.

        Args:
            scheduled: Set of already-scheduled instruction IDs.

        Returns:
            List of instruction IDs ready to be scheduled next.
        """
        ready = []
        for inst_id in range(self.n):
            if inst_id in scheduled:
                continue
            if self.predecessors[inst_id].issubset(scheduled):
                ready.append(inst_id)
        return ready

    def get_edge_index(self) -> List[Tuple[int, int]]:
        """Return edges as COO list [(src, dst), ...] for GNN consumption."""
        seen = set()
        result = []
        for edge in self.edges:
            pair = (edge.src, edge.dst)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
        return result

    def get_edge_types(self) -> Dict[Tuple[int, int], List[DepType]]:
        """Return dependency types for each edge pair."""
        result = defaultdict(list)
        for edge in self.edges:
            dep = edge.dep_type
            if dep not in result[(edge.src, edge.dst)]:
                result[(edge.src, edge.dst)].append(dep)
        return dict(result)

    def critical_path_instructions(self) -> List[int]:
        """Return instruction IDs along the critical path (longest weighted path)."""
        if not self.instructions:
            return []

        # Find the leaf with maximum depth + own latency
        max_total = -1
        end_node = -1
        for leaf_id in self.leaves:
            total = self.depth[leaf_id] + self.instructions[leaf_id].latency
            if total > max_total:
                max_total = total
                end_node = leaf_id

        if end_node == -1:
            return []

        # Trace back from end_node to a root following the critical path
        path = [end_node]
        current = end_node
        while self.predecessors[current]:
            # Pick the predecessor with the highest depth
            best_pred = max(
                self.predecessors[current],
                key=lambda p: self.depth[p] + self.instructions[p].latency
            )
            path.append(best_pred)
            current = best_pred

        return list(reversed(path))

    def __repr__(self):
        return (
            f"InstructionDAG(nodes={self.n}, edges={len(self.edges)}, "
            f"critical_path={self.critical_path_length} cycles)"
        )
